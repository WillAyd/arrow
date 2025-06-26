// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

#include <gtest/gtest.h>

#include "arrow/array/array_primitive.h"
#include "arrow/array/builder_base.h"
#include "arrow/array/builder_primitive.h"
#include "arrow/memory_pool.h"
#include "arrow/testing/builder.h"
#include "arrow/testing/gtest_util.h"
#include "arrow/testing/random.h"
#include "arrow/testing/util.h"
#include "arrow/type.h"
#include "arrow/util/bitmap_builders.h"
#include "arrow/util/checked_cast.h"
#include "arrow/util/range.h"

namespace arrow {

using internal::checked_cast;
using internal::checked_pointer_cast;

template <typename Attrs>
class TestPrimitiveBuilder : public ::testing::Test {
 public:
  typedef Attrs TestAttrs;
  typedef typename Attrs::ArrayType ArrayType;
  typedef typename Attrs::BuilderType BuilderType;
  typedef typename Attrs::T CType;
  typedef typename Attrs::Type Type;

  virtual void SetUp() {
    type_ = Attrs::type();

    std::unique_ptr<ArrayBuilder> tmp;
    ASSERT_OK(MakeBuilder(pool_, type_, &tmp));
    builder_.reset(checked_cast<BuilderType*>(tmp.release()));

    ASSERT_OK(MakeBuilder(pool_, type_, &tmp));
    builder_nn_.reset(checked_cast<BuilderType*>(tmp.release()));
  }

  void RandomData(int64_t N, double pct_null = 0.1) {
    Attrs::draw(N, &draws_);

    valid_bytes_.resize(static_cast<size_t>(N));
    random_null_bytes(N, pct_null, valid_bytes_.data());
  }

  void Check(const std::unique_ptr<BuilderType>& builder, bool nullable) {
    int64_t size = builder->length();
    auto ex_data = Buffer::Wrap(draws_.data(), size);

    std::shared_ptr<Buffer> ex_null_bitmap;
    int64_t ex_null_count = 0;

    if (nullable) {
      ASSERT_OK_AND_ASSIGN(ex_null_bitmap, internal::BytesToBits(valid_bytes_));
      ex_null_count = CountNulls(valid_bytes_);
    } else {
      ex_null_bitmap = nullptr;
    }

    auto expected =
        std::make_shared<ArrayType>(size, ex_data, ex_null_bitmap, ex_null_count);

    std::shared_ptr<Array> out;
    FinishAndCheckPadding(builder.get(), &out);

    std::shared_ptr<ArrayType> result = checked_pointer_cast<ArrayType>(out);

    // Builder is now reset
    ASSERT_EQ(0, builder->length());
    ASSERT_EQ(0, builder->capacity());
    ASSERT_EQ(0, builder->null_count());

    ASSERT_EQ(ex_null_count, result->null_count());
    ASSERT_TRUE(result->Equals(*expected));
  }

  void FlipValue(CType* ptr) {
    auto byteptr = reinterpret_cast<uint8_t*>(ptr);
    *byteptr = static_cast<uint8_t>(~*byteptr);
  }

 protected:
  MemoryPool* pool_ = default_memory_pool();
  std::shared_ptr<DataType> type_;

  std::unique_ptr<BuilderType> builder_;
  std::unique_ptr<BuilderType> builder_nn_;

  std::vector<CType> draws_;
  std::vector<uint8_t> valid_bytes_;
};

/// \brief uint8_t isn't a valid template parameter to uniform_int_distribution, so
/// we use SampleType to determine which kind of integer to use to sample.
template <typename T, typename = enable_if_t<std::is_integral<T>::value, T>>
struct UniformIntSampleType {
  using type = T;
};

template <>
struct UniformIntSampleType<uint8_t> {
  using type = uint16_t;
};

template <>
struct UniformIntSampleType<int8_t> {
  using type = int16_t;
};

#define PTYPE_DECL(CapType, c_type)     \
  typedef CapType##Array ArrayType;     \
  typedef CapType##Builder BuilderType; \
  typedef CapType##Type Type;           \
  typedef c_type T;                     \
                                        \
  static std::shared_ptr<DataType> type() { return std::make_shared<Type>(); }

#define PINT_DECL(CapType, c_type)                                                     \
  struct P##CapType {                                                                  \
    PTYPE_DECL(CapType, c_type)                                                        \
    static void draw(int64_t N, std::vector<T>* draws) {                               \
      using sample_type = typename UniformIntSampleType<c_type>::type;                 \
      const T lower = std::numeric_limits<T>::min();                                   \
      const T upper = std::numeric_limits<T>::max();                                   \
      randint(N, static_cast<sample_type>(lower), static_cast<sample_type>(upper),     \
              draws);                                                                  \
    }                                                                                  \
    static T Modify(T inp) { return inp / 2; }                                         \
    typedef                                                                            \
        typename std::conditional<std::is_unsigned<T>::value, uint64_t, int64_t>::type \
            ConversionType;                                                            \
  }

#define PFLOAT_DECL(CapType, c_type, LOWER, UPPER)       \
  struct P##CapType {                                    \
    PTYPE_DECL(CapType, c_type)                          \
    static void draw(int64_t N, std::vector<T>* draws) { \
      random_real(N, 0, LOWER, UPPER, draws);            \
    }                                                    \
    static T Modify(T inp) { return inp / 2; }           \
    typedef double ConversionType;                       \
  }

PINT_DECL(UInt8, uint8_t);
PINT_DECL(UInt16, uint16_t);
PINT_DECL(UInt32, uint32_t);
PINT_DECL(UInt64, uint64_t);

PINT_DECL(Int8, int8_t);
PINT_DECL(Int16, int16_t);
PINT_DECL(Int32, int32_t);
PINT_DECL(Int64, int64_t);

PFLOAT_DECL(Float, float, -1000.0f, 1000.0f);
PFLOAT_DECL(Double, double, -1000.0, 1000.0);

struct PBoolean {
  PTYPE_DECL(Boolean, uint8_t)
  static T Modify(T inp) { return !inp; }
  typedef int64_t ConversionType;
};

struct PDayTimeInterval {
  using DayMilliseconds = DayTimeIntervalType::DayMilliseconds;
  PTYPE_DECL(DayTimeInterval, DayMilliseconds);
  static void draw(int64_t N, std::vector<T>* draws) { return rand_day_millis(N, draws); }

  static DayMilliseconds Modify(DayMilliseconds inp) {
    inp.days /= 2;
    return inp;
  }
  typedef DayMilliseconds ConversionType;
};

struct PMonthDayNanoInterval {
  using MonthDayNanos = MonthDayNanoIntervalType::MonthDayNanos;
  PTYPE_DECL(MonthDayNanoInterval, MonthDayNanos);
  static void draw(int64_t N, std::vector<T>* draws) {
    return rand_month_day_nanos(N, draws);
  }
  static MonthDayNanos Modify(MonthDayNanos inp) {
    inp.days /= 2;
    return inp;
  }
  typedef MonthDayNanos ConversionType;
};

template <>
void TestPrimitiveBuilder<PBoolean>::RandomData(int64_t N, double pct_null) {
  draws_.resize(static_cast<size_t>(N));
  valid_bytes_.resize(static_cast<size_t>(N));

  random_null_bytes(N, 0.5, draws_.data());
  random_null_bytes(N, pct_null, valid_bytes_.data());
}

template <>
void TestPrimitiveBuilder<PBoolean>::FlipValue(CType* ptr) {
  *ptr = !*ptr;
}

template <>
void TestPrimitiveBuilder<PBoolean>::Check(const std::unique_ptr<BooleanBuilder>& builder,
                                           bool nullable) {
  const int64_t size = builder->length();

  // Build expected result array
  std::shared_ptr<Buffer> ex_data;
  std::shared_ptr<Buffer> ex_null_bitmap;
  int64_t ex_null_count = 0;

  ASSERT_OK_AND_ASSIGN(ex_data, internal::BytesToBits(draws_));
  if (nullable) {
    ASSERT_OK_AND_ASSIGN(ex_null_bitmap, internal::BytesToBits(valid_bytes_));
    ex_null_count = CountNulls(valid_bytes_);
  } else {
    ex_null_bitmap = nullptr;
  }
  auto expected =
      std::make_shared<BooleanArray>(size, ex_data, ex_null_bitmap, ex_null_count);
  ASSERT_EQ(size, expected->length());

  // Finish builder and check result array
  std::shared_ptr<Array> out;
  FinishAndCheckPadding(builder.get(), &out);

  std::shared_ptr<BooleanArray> result = checked_pointer_cast<BooleanArray>(out);

  ASSERT_EQ(ex_null_count, result->null_count());
  ASSERT_EQ(size, result->length());

  for (int64_t i = 0; i < size; ++i) {
    if (nullable) {
      ASSERT_EQ(valid_bytes_[i] == 0, result->IsNull(i)) << i;
    } else {
      ASSERT_FALSE(result->IsNull(i));
    }
    if (!result->IsNull(i)) {
      bool actual = bit_util::GetBit(result->values()->data(), i);
      ASSERT_EQ(draws_[i] != 0, actual) << i;
    }
  }
  AssertArraysEqual(*result, *expected);

  // buffers are correctly sized
  if (result->data()->buffers[0]) {
    ASSERT_EQ(result->data()->buffers[0]->size(), bit_util::BytesForBits(size));
  } else {
    ASSERT_EQ(result->data()->null_count, 0);
  }
  ASSERT_EQ(result->data()->buffers[1]->size(), bit_util::BytesForBits(size));

  // Builder is now reset
  ASSERT_EQ(0, builder->length());
  ASSERT_EQ(0, builder->capacity());
  ASSERT_EQ(0, builder->null_count());
}

typedef ::testing::Types<PBoolean, PUInt8, PUInt16, PUInt32, PUInt64, PInt8, PInt16,
                         PInt32, PInt64, PFloat, PDouble, PDayTimeInterval,
                         PMonthDayNanoInterval>
    Primitives;

TYPED_TEST_SUITE(TestPrimitiveBuilder, Primitives);

TYPED_TEST(TestPrimitiveBuilder, TestInit) {
  ASSERT_OK(this->builder_->Reserve(1000));
  ASSERT_EQ(1000, this->builder_->capacity());

  // Small upsize => should overallocate
  ASSERT_OK(this->builder_->Reserve(1200));
  ASSERT_GE(2000, this->builder_->capacity());

  // Large upsize => should allocate exactly
  ASSERT_OK(this->builder_->Reserve(32768));
  ASSERT_EQ(32768, this->builder_->capacity());

  // unsure if this should go in all builder classes
  ASSERT_EQ(0, this->builder_->num_children());
}

TYPED_TEST(TestPrimitiveBuilder, TestAppendNull) {
  int64_t size = 1000;
  for (int64_t i = 0; i < size; ++i) {
    ASSERT_OK(this->builder_->AppendNull());
    ASSERT_EQ(i + 1, this->builder_->null_count());
  }

  std::shared_ptr<Array> out;
  FinishAndCheckPadding(this->builder_.get(), &out);
  auto result = checked_pointer_cast<typename TypeParam::ArrayType>(out);

  for (int64_t i = 0; i < size; ++i) {
    ASSERT_TRUE(result->IsNull(i)) << i;
  }

  for (auto buffer : result->data()->buffers) {
    for (int64_t i = 0; i < buffer->capacity(); i++) {
      // Validates current implementation, algorithms shouldn't rely on this
      ASSERT_EQ(0, *(buffer->data() + i)) << i;
    }
  }
}

TYPED_TEST(TestPrimitiveBuilder, TestAppendOptional) {
  int64_t size = 1000;
  for (int64_t i = 0; i < size; ++i) {
    ASSERT_OK(this->builder_->AppendOrNull(std::nullopt));
    ASSERT_EQ(i + 1, this->builder_->null_count());
  }

  std::shared_ptr<Array> out;
  FinishAndCheckPadding(this->builder_.get(), &out);
  auto result = checked_pointer_cast<typename TypeParam::ArrayType>(out);

  for (int64_t i = 0; i < size; ++i) {
    ASSERT_TRUE(result->IsNull(i)) << i;
  }
}

TYPED_TEST(TestPrimitiveBuilder, TestAppendNulls) {
  const int64_t size = 10;
  ASSERT_OK(this->builder_->AppendNulls(size));
  ASSERT_EQ(size, this->builder_->null_count());

  std::shared_ptr<Array> result;
  FinishAndCheckPadding(this->builder_.get(), &result);

  for (int64_t i = 0; i < size; ++i) {
    ASSERT_FALSE(result->IsValid(i));
  }

  for (auto buffer : result->data()->buffers) {
    for (int64_t i = 0; i < buffer->capacity(); i++) {
      // Validates current implementation, algorithms shouldn't rely on this
      ASSERT_EQ(0, *(buffer->data() + i)) << i;
    }
  }
}

TYPED_TEST(TestPrimitiveBuilder, TestAppendEmptyValue) {
  ASSERT_OK(this->builder_->AppendNull());
  ASSERT_OK(this->builder_->AppendEmptyValue());
  ASSERT_OK(this->builder_->AppendNulls(2));
  ASSERT_OK(this->builder_->AppendEmptyValues(2));

  std::shared_ptr<Array> out;
  FinishAndCheckPadding(this->builder_.get(), &out);
  ASSERT_OK(out->ValidateFull());

  auto result = checked_pointer_cast<typename TypeParam::ArrayType>(out);
  ASSERT_EQ(result->length(), 6);
  ASSERT_EQ(result->null_count(), 3);

  ASSERT_TRUE(result->IsNull(0));
  ASSERT_FALSE(result->IsNull(1));
  ASSERT_TRUE(result->IsNull(2));
  ASSERT_TRUE(result->IsNull(3));
  ASSERT_FALSE(result->IsNull(4));
  ASSERT_FALSE(result->IsNull(5));

  // implementation detail: the value slots are 0-initialized
  for (int64_t i = 0; i < result->length(); ++i) {
    typename TestFixture::CType t{};
    ASSERT_EQ(result->Value(i), t);
  }
}

TYPED_TEST(TestPrimitiveBuilder, TestArrayDtorDealloc) {
  typedef typename TestFixture::CType T;

  int64_t size = 1000;

  std::vector<T>& draws = this->draws_;
  std::vector<uint8_t>& valid_bytes = this->valid_bytes_;

  int64_t memory_before = this->pool_->bytes_allocated();

  this->RandomData(size);
  ASSERT_OK(this->builder_->Reserve(size));

  int64_t i;
  for (i = 0; i < size; ++i) {
    if (valid_bytes[i] > 0) {
      ASSERT_OK(this->builder_->Append(draws[i]));
    } else {
      ASSERT_OK(this->builder_->AppendNull());
    }
  }

  do {
    std::shared_ptr<Array> result;
    FinishAndCheckPadding(this->builder_.get(), &result);
  } while (false);

  ASSERT_EQ(memory_before, this->pool_->bytes_allocated());
}

TYPED_TEST(TestPrimitiveBuilder, Equality) {
  typedef typename TestFixture::CType T;

  const int64_t size = 1000;
  this->RandomData(size);
  std::vector<T>& draws = this->draws_;
  std::vector<uint8_t>& valid_bytes = this->valid_bytes_;
  std::shared_ptr<Array> array, equal_array, unequal_array;
  auto builder = this->builder_.get();
  ASSERT_OK(MakeArray(valid_bytes, draws, size, builder, &array));
  ASSERT_OK(MakeArray(valid_bytes, draws, size, builder, &equal_array));

  // Make the not equal array by negating the first valid element with itself.
  const auto first_valid = std::find_if(valid_bytes.begin(), valid_bytes.end(),
                                        [](uint8_t valid) { return valid > 0; });
  const int64_t first_valid_idx = std::distance(valid_bytes.begin(), first_valid);
  // This should be true with a very high probability, but might introduce flakiness
  ASSERT_LT(first_valid_idx, size - 1);
  this->FlipValue(&draws[first_valid_idx]);
  ASSERT_OK(MakeArray(valid_bytes, draws, size, builder, &unequal_array));

  // test normal equality
  EXPECT_TRUE(array->Equals(array));
  EXPECT_TRUE(array->Equals(equal_array));
  EXPECT_TRUE(equal_array->Equals(array));
  EXPECT_FALSE(equal_array->Equals(unequal_array));
  EXPECT_FALSE(unequal_array->Equals(equal_array));

  // Test range equality
  EXPECT_FALSE(array->RangeEquals(0, first_valid_idx + 1, 0, unequal_array));
  EXPECT_FALSE(array->RangeEquals(first_valid_idx, size, first_valid_idx, unequal_array));
  EXPECT_TRUE(array->RangeEquals(0, first_valid_idx, 0, unequal_array));
  EXPECT_TRUE(
      array->RangeEquals(first_valid_idx + 1, size, first_valid_idx + 1, unequal_array));
}

TYPED_TEST(TestPrimitiveBuilder, SliceEquality) {
  typedef typename TestFixture::CType T;

  const int64_t size = 1000;
  this->RandomData(size);
  std::vector<T>& draws = this->draws_;
  std::vector<uint8_t>& valid_bytes = this->valid_bytes_;
  auto builder = this->builder_.get();

  std::shared_ptr<Array> array;
  ASSERT_OK(MakeArray(valid_bytes, draws, size, builder, &array));

  std::shared_ptr<Array> slice, slice2;

  slice = array->Slice(5);
  slice2 = array->Slice(5);
  ASSERT_EQ(size - 5, slice->length());

  ASSERT_TRUE(slice->Equals(slice2));
  ASSERT_TRUE(array->RangeEquals(5, array->length(), 0, slice));

  // Chained slices
  slice2 = array->Slice(2)->Slice(3);
  ASSERT_TRUE(slice->Equals(slice2));

  slice = array->Slice(5, 10);
  slice2 = array->Slice(5, 10);
  ASSERT_EQ(10, slice->length());

  ASSERT_TRUE(slice->Equals(slice2));
  ASSERT_TRUE(array->RangeEquals(5, 15, 0, slice));
}

TYPED_TEST(TestPrimitiveBuilder, TestAppendScalar) {
  typedef typename TestFixture::CType T;

  const int64_t size = 10000;

  std::vector<T>& draws = this->draws_;
  std::vector<uint8_t>& valid_bytes = this->valid_bytes_;

  this->RandomData(size);

  ASSERT_OK(this->builder_->Reserve(1000));
  ASSERT_OK(this->builder_nn_->Reserve(1000));

  int64_t null_count = 0;
  // Append the first 1000
  for (size_t i = 0; i < 1000; ++i) {
    if (valid_bytes[i] > 0) {
      ASSERT_OK(this->builder_->Append(draws[i]));
    } else {
      ASSERT_OK(this->builder_->AppendNull());
      ++null_count;
    }
    ASSERT_OK(this->builder_nn_->Append(draws[i]));
  }

  ASSERT_EQ(null_count, this->builder_->null_count());

  ASSERT_EQ(1000, this->builder_->length());
  ASSERT_EQ(1000, this->builder_->capacity());

  ASSERT_EQ(1000, this->builder_nn_->length());
  ASSERT_EQ(1000, this->builder_nn_->capacity());

  ASSERT_OK(this->builder_->Reserve(size - 1000));
  ASSERT_OK(this->builder_nn_->Reserve(size - 1000));

  // Append the next 9000
  for (size_t i = 1000; i < size; ++i) {
    if (valid_bytes[i] > 0) {
      ASSERT_OK(this->builder_->Append(draws[i]));
    } else {
      ASSERT_OK(this->builder_->AppendNull());
    }
    ASSERT_OK(this->builder_nn_->Append(draws[i]));
  }

  ASSERT_EQ(size, this->builder_->length());
  ASSERT_GE(size, this->builder_->capacity());

  ASSERT_EQ(size, this->builder_nn_->length());
  ASSERT_GE(size, this->builder_nn_->capacity());

  this->Check(this->builder_, true);
  this->Check(this->builder_nn_, false);
}

TYPED_TEST(TestPrimitiveBuilder, TestAppendValues) {
  typedef typename TestFixture::CType T;

  int64_t size = 10000;
  this->RandomData(size);

  std::vector<T>& draws = this->draws_;
  std::vector<uint8_t>& valid_bytes = this->valid_bytes_;

  // first slug
  int64_t K = 1000;

  ASSERT_OK(this->builder_->AppendValues(draws.data(), K, valid_bytes.data()));
  ASSERT_OK(this->builder_nn_->AppendValues(draws.data(), K));

  ASSERT_EQ(1000, this->builder_->length());
  ASSERT_EQ(1000, this->builder_->capacity());

  ASSERT_EQ(1000, this->builder_nn_->length());
  ASSERT_EQ(1000, this->builder_nn_->capacity());

  // Append the next 9000
  ASSERT_OK(
      this->builder_->AppendValues(draws.data() + K, size - K, valid_bytes.data() + K));
  ASSERT_OK(this->builder_nn_->AppendValues(draws.data() + K, size - K));

  ASSERT_EQ(size, this->builder_->length());
  ASSERT_GE(size, this->builder_->capacity());

  ASSERT_EQ(size, this->builder_nn_->length());
  ASSERT_GE(size, this->builder_nn_->capacity());

  this->Check(this->builder_, true);
  this->Check(this->builder_nn_, false);
}

TYPED_TEST(TestPrimitiveBuilder, TestTypedFinish) {
  typedef typename TestFixture::CType T;

  int64_t size = 1000;
  this->RandomData(size);

  std::vector<T>& draws = this->draws_;
  std::vector<uint8_t>& valid_bytes = this->valid_bytes_;

  ASSERT_OK(this->builder_->AppendValues(draws.data(), size, valid_bytes.data()));
  std::shared_ptr<Array> result_untyped;
  ASSERT_OK(this->builder_->Finish(&result_untyped));

  ASSERT_OK(this->builder_->AppendValues(draws.data(), size, valid_bytes.data()));
  std::shared_ptr<typename TestFixture::ArrayType> result;
  ASSERT_OK(this->builder_->Finish(&result));

  AssertArraysEqual(*result_untyped, *result);
}

TYPED_TEST(TestPrimitiveBuilder, TestAppendValuesIter) {
  int64_t size = 10000;
  this->RandomData(size);

  ASSERT_OK(this->builder_->AppendValues(this->draws_.begin(), this->draws_.end(),
                                         this->valid_bytes_.begin()));
  ASSERT_OK(this->builder_nn_->AppendValues(this->draws_.begin(), this->draws_.end()));

  ASSERT_EQ(size, this->builder_->length());
  ASSERT_GE(size, this->builder_->capacity());

  this->Check(this->builder_, true);
  this->Check(this->builder_nn_, false);
}

TYPED_TEST(TestPrimitiveBuilder, TestAppendValuesIterNullValid) {
  int64_t size = 10000;
  this->RandomData(size);

  ASSERT_OK(this->builder_nn_->AppendValues(this->draws_.begin(),
                                            this->draws_.begin() + size / 2,
                                            static_cast<uint8_t*>(nullptr)));

  ASSERT_GE(size / 2, this->builder_nn_->capacity());

  ASSERT_OK(this->builder_nn_->AppendValues(this->draws_.begin() + size / 2,
                                            this->draws_.end(),
                                            static_cast<uint64_t*>(nullptr)));

  this->Check(this->builder_nn_, false);
}

TYPED_TEST(TestPrimitiveBuilder, TestAppendValuesLazyIter) {
  typedef typename TestFixture::CType T;

  int64_t size = 10000;
  this->RandomData(size);

  auto& draws = this->draws_;
  auto& valid_bytes = this->valid_bytes_;

  auto halve = [&draws](int64_t index) {
    return TestFixture::TestAttrs::Modify(draws[index]);
  };
  auto lazy_iter = internal::MakeLazyRange(halve, size);

  ASSERT_OK(this->builder_->AppendValues(lazy_iter.begin(), lazy_iter.end(),
                                         valid_bytes.begin()));

  std::vector<T> halved;
  transform(draws.begin(), draws.end(), back_inserter(halved),
            [](T in) { return TestFixture::TestAttrs::Modify(in); });

  std::shared_ptr<Array> result;
  FinishAndCheckPadding(this->builder_.get(), &result);

  std::shared_ptr<Array> expected;
  ASSERT_OK(
      this->builder_->AppendValues(halved.data(), halved.size(), valid_bytes.data()));
  FinishAndCheckPadding(this->builder_.get(), &expected);

  ASSERT_TRUE(expected->Equals(result));
}

TYPED_TEST(TestPrimitiveBuilder, TestAppendValuesIterConverted) {
  typedef typename TestFixture::CType T;
  // find type we can safely convert the tested values to and from
  using conversion_type = typename TestFixture::TestAttrs::ConversionType;

  int64_t size = 10000;
  this->RandomData(size);

  // append convertible values
  std::vector<conversion_type> draws_converted(this->draws_.begin(), this->draws_.end());
  std::vector<int32_t> valid_bytes_converted(this->valid_bytes_.begin(),
                                             this->valid_bytes_.end());

  auto cast_values = internal::MakeLazyRange(
      [&draws_converted](int64_t index) {
        return static_cast<T>(draws_converted[index]);
      },
      size);
  auto cast_valid = internal::MakeLazyRange(
      [&valid_bytes_converted](int64_t index) {
        return static_cast<bool>(valid_bytes_converted[index]);
      },
      size);

  ASSERT_OK(this->builder_->AppendValues(cast_values.begin(), cast_values.end(),
                                         cast_valid.begin()));
  ASSERT_OK(this->builder_nn_->AppendValues(cast_values.begin(), cast_values.end()));

  ASSERT_EQ(size, this->builder_->length());
  ASSERT_GE(size, this->builder_->capacity());

  ASSERT_EQ(size, this->builder_->length());
  ASSERT_GE(size, this->builder_->capacity());

  this->Check(this->builder_, true);
  this->Check(this->builder_nn_, false);
}

TYPED_TEST(TestPrimitiveBuilder, TestZeroPadded) {
  typedef typename TestFixture::CType T;

  int64_t size = 10000;
  this->RandomData(size);

  std::vector<T>& draws = this->draws_;
  std::vector<uint8_t>& valid_bytes = this->valid_bytes_;

  // first slug
  int64_t K = 1000;

  ASSERT_OK(this->builder_->AppendValues(draws.data(), K, valid_bytes.data()));

  std::shared_ptr<Array> out;
  FinishAndCheckPadding(this->builder_.get(), &out);
}

TYPED_TEST(TestPrimitiveBuilder, TestAppendValuesStdBool) {
  // ARROW-1383
  typedef typename TestFixture::CType T;

  int64_t size = 10000;
  this->RandomData(size);

  std::vector<T>& draws = this->draws_;

  std::vector<bool> is_valid;

  // first slug
  int64_t K = 1000;

  for (int64_t i = 0; i < K; ++i) {
    is_valid.push_back(this->valid_bytes_[i] != 0);
  }
  ASSERT_OK(this->builder_->AppendValues(draws.data(), K, is_valid));
  ASSERT_OK(this->builder_nn_->AppendValues(draws.data(), K));

  ASSERT_EQ(1000, this->builder_->length());
  ASSERT_EQ(1000, this->builder_->capacity());
  ASSERT_EQ(1000, this->builder_nn_->length());
  ASSERT_EQ(1000, this->builder_nn_->capacity());

  // Append the next 9000
  is_valid.clear();
  std::vector<T> partial_draws;
  for (int64_t i = K; i < size; ++i) {
    partial_draws.push_back(draws[i]);
    is_valid.push_back(this->valid_bytes_[i] != 0);
  }

  ASSERT_OK(this->builder_->AppendValues(partial_draws, is_valid));
  ASSERT_OK(this->builder_nn_->AppendValues(partial_draws));

  ASSERT_EQ(size, this->builder_->length());
  ASSERT_GE(size, this->builder_->capacity());

  ASSERT_EQ(size, this->builder_nn_->length());
  ASSERT_GE(size, this->builder_->capacity());

  this->Check(this->builder_, true);
  this->Check(this->builder_nn_, false);
}

TYPED_TEST(TestPrimitiveBuilder, TestResize) {
  int64_t cap = kMinBuilderCapacity * 2;

  ASSERT_OK(this->builder_->Reserve(cap));
  ASSERT_EQ(cap, this->builder_->capacity());
}

TYPED_TEST(TestPrimitiveBuilder, TestReserve) {
  ASSERT_OK(this->builder_->Reserve(10));
  ASSERT_EQ(0, this->builder_->length());
  ASSERT_EQ(kMinBuilderCapacity, this->builder_->capacity());

  ASSERT_OK(this->builder_->Reserve(100));
  ASSERT_EQ(0, this->builder_->length());
  ASSERT_GE(100, this->builder_->capacity());
  ASSERT_OK(this->builder_->AppendEmptyValues(100));
  ASSERT_EQ(100, this->builder_->length());
  ASSERT_GE(100, this->builder_->capacity());

  ASSERT_RAISES(Invalid, this->builder_->Resize(1));
}

template <typename PType>
class TestPrimitiveArray : public ::testing::Test {
 public:
  using ElementType = typename PType::T;

  void SetUp() {
    pool_ = default_memory_pool();
    GenerateInput();
  }

  void GenerateInput() {
    validity_ = std::vector<bool>{true, false, true, true, false, true};
    values_ = std::vector<ElementType>{0, 1, 1, 0, 1, 1};
  }

 protected:
  MemoryPool* pool_;
  std::vector<bool> validity_;
  std::vector<ElementType> values_;
};

template <>
void TestPrimitiveArray<PDayTimeInterval>::GenerateInput() {
  validity_ = std::vector<bool>{true, false};
  values_ = std::vector<DayTimeIntervalType::DayMilliseconds>{{0, 10}, {1, 0}};
}

template <>
void TestPrimitiveArray<PMonthDayNanoInterval>::GenerateInput() {
  validity_ = std::vector<bool>{false, true};
  values_ =
      std::vector<MonthDayNanoIntervalType::MonthDayNanos>{{0, 10, 100}, {1, 0, 10}};
}

TYPED_TEST_SUITE(TestPrimitiveArray, Primitives);

TYPED_TEST(TestPrimitiveArray, IndexOperator) {
  typename TypeParam::BuilderType builder;
  ASSERT_OK(builder.Reserve(this->values_.size()));
  ASSERT_OK(builder.AppendValues(this->values_, this->validity_));
  ASSERT_OK_AND_ASSIGN(auto array, builder.Finish());

  const auto& carray = checked_cast<typename TypeParam::ArrayType&>(*array);

  ASSERT_EQ(this->values_.size(), carray.length());
  for (int64_t i = 0; i < carray.length(); ++i) {
    auto res = carray[i];
    if (this->validity_[i]) {
      ASSERT_TRUE(res.has_value());
      ASSERT_EQ(this->values_[i], res.value());
    } else {
      ASSERT_FALSE(res.has_value());
      ASSERT_EQ(res, std::nullopt);
    }
  }
}
}  // namespace arrow
