use std::sync::Arc;

use serde::{
    de::{self, Visitor},
    ser::SerializeStruct,
    Deserialize, Serialize,
};

use crate::Tensor;

const FIELDS: &[&str] = &["shape", "data"];

// Adapted from https://serde.rs/impl-serialize.html
impl Serialize for Tensor {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        // Serialize tensor
        let mut state = serializer.serialize_struct("Tensor", FIELDS.len())?;
        state.serialize_field(FIELDS[0], &self.shape)?;
        state.serialize_field(FIELDS[1], &*self.data)?;
        state.end()
    }
}

// Adapted from https://serde.rs/deserialize-struct.html
impl<'de> Deserialize<'de> for Tensor {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        enum Field {
            Shape,
            Data,
        }

        impl<'de> Deserialize<'de> for Field {
            fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
            where
                D: serde::Deserializer<'de>,
            {
                struct FieldVisitor;

                impl Visitor<'_> for FieldVisitor {
                    type Value = Field;

                    fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
                        formatter.write_str("field identifier")
                    }

                    fn visit_str<E>(self, value: &str) -> Result<Self::Value, E>
                    where
                        E: serde::de::Error,
                    {
                        match value {
                            "shape" => Ok(Field::Shape),
                            "data" => Ok(Field::Data),
                            _ => Err(serde::de::Error::unknown_field(value, FIELDS)),
                        }
                    }
                }

                deserializer.deserialize_identifier(FieldVisitor)
            }
        }

        struct TensorVisitor;

        impl<'de> Visitor<'de> for TensorVisitor {
            type Value = Tensor;

            fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
                formatter.write_str("struct Tensor")
            }

            fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
            where
                A: de::SeqAccess<'de>,
            {
                let shape = seq
                    .next_element()?
                    .ok_or_else(|| de::Error::invalid_length(0, &self))?;
                let data = seq
                    .next_element()?
                    .ok_or_else(|| de::Error::invalid_length(2, &self))?;

                // Create the tensor
                Ok(Tensor {
                    shape,
                    data: Arc::new(data),
                })
            }

            fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
            where
                A: serde::de::MapAccess<'de>,
            {
                let mut shape = None;
                let mut data = None;
                while let Some(key) = map.next_key()? {
                    match key {
                        Field::Shape => {
                            if shape.is_some() {
                                return Err(serde::de::Error::duplicate_field("shape"));
                            }
                            shape = Some(map.next_value()?);
                        }
                        Field::Data => {
                            if data.is_some() {
                                return Err(serde::de::Error::duplicate_field("data"));
                            }
                            data = Some(map.next_value()?);
                        }
                    }
                }

                // Unpack the fields
                let shape = shape.ok_or_else(|| de::Error::missing_field("shape"))?;
                let data = data.ok_or_else(|| de::Error::missing_field("data"))?;

                // Create the tensor
                Ok(Tensor {
                    shape,
                    data: Arc::new(data),
                })
            }
        }

        deserializer.deserialize_struct("Tensor", FIELDS, TensorVisitor)
    }
}

#[cfg(test)]
mod tests {
    use num_complex::Complex64;
    use serde::Serialize;

    use crate::Tensor;
    use serde_test::{assert_tokens, Token};

    #[derive(Debug)]
    struct TensorEqWrapper(Tensor);

    impl PartialEq for TensorEqWrapper {
        fn eq(&self, other: &Self) -> bool {
            self.0.shape == other.0.shape && self.0.data == other.0.data
        }
    }

    impl Serialize for TensorEqWrapper {
        fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
        where
            S: serde::Serializer,
        {
            self.0.serialize(serializer)
        }
    }

    impl<'de> serde::Deserialize<'de> for TensorEqWrapper {
        fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
        where
            D: serde::Deserializer<'de>,
        {
            Ok(TensorEqWrapper(Tensor::deserialize(deserializer)?))
        }
    }

    #[test]
    fn test_serde_scalar() {
        let tensor = Tensor::new_scalar(Complex64::new(1.0, 2.0));

        assert_tokens(
            &TensorEqWrapper(tensor),
            &[
                serde_test::Token::Struct {
                    name: "Tensor",
                    len: 2,
                },
                Token::Str("shape"),
                Token::Seq { len: Some(0) },
                Token::SeqEnd,
                Token::Str("data"),
                Token::Seq { len: Some(1) },
                Token::Tuple { len: 2 },
                Token::F64(1.0),
                Token::F64(2.0),
                Token::TupleEnd,
                Token::SeqEnd,
                Token::StructEnd,
            ],
        );
    }

    #[test]
    fn test_serde_simple() {
        let a_data = vec![
            Complex64::new(1.0, 1.0),
            Complex64::new(-2.0, 0.0),
            Complex64::new(-3.0, -1.0),
            Complex64::new(0.0, 5.0),
        ];
        let tensor = Tensor::new_from_flat(&[2, 2, 1], a_data);

        // Hint: Serde always serializes usize as U64
        assert_tokens(
            &TensorEqWrapper(tensor),
            &[
                serde_test::Token::Struct {
                    name: "Tensor",
                    len: 2,
                },
                Token::Str("shape"),
                Token::Seq { len: Some(3) },
                Token::U64(2),
                Token::U64(2),
                Token::U64(1),
                Token::SeqEnd,
                Token::Str("data"),
                Token::Seq { len: Some(4) },
                Token::Tuple { len: 2 },
                Token::F64(1.0),
                Token::F64(1.0),
                Token::TupleEnd,
                Token::Tuple { len: 2 },
                Token::F64(-2.0),
                Token::F64(0.0),
                Token::TupleEnd,
                Token::Tuple { len: 2 },
                Token::F64(-3.0),
                Token::F64(-1.0),
                Token::TupleEnd,
                Token::Tuple { len: 2 },
                Token::F64(0.0),
                Token::F64(5.0),
                Token::TupleEnd,
                Token::SeqEnd,
                Token::StructEnd,
            ],
        );
    }
}
