use std::{cell::RefCell, ops::Deref, rc::Rc};

use num_complex::Complex64;
use permutation::Permutation;
use serde::{
    de::{self, Visitor},
    ser::SerializeStruct,
    Deserialize, Serialize,
};

use crate::Tensor;

const FIELDS: &[&str] = &["shape", "permutation", "data"];

/// Converts a permutation to a vector that can be serialized.
/// The vector is in zero-based oneline notation (see [`Permutation::oneline`]).
fn permutation_to_raw<T>(perm: T) -> Vec<usize>
where
    T: Deref<Target = Permutation>,
{
    let normalized = (*perm).clone().normalize(false);
    (0..normalized.len())
        .map(|idx| normalized.apply_idx(idx))
        .collect()
}

/// Converts a vector in zero-based oneline notation to a permutation.
fn raw_to_permutation(raw: &[usize]) -> Permutation {
    Permutation::oneline(raw.to_vec())
}

// Adapted from https://serde.rs/impl-serialize.html
impl Serialize for Tensor {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let shape = self.shape.borrow();
        let permutation = self.permutation.borrow();
        let data = self.data.borrow();

        // Serialize tensor
        let mut state = serializer.serialize_struct("Tensor", FIELDS.len())?;
        state.serialize_field(FIELDS[0], &shape.as_slice())?;
        state.serialize_field(FIELDS[1], &permutation_to_raw(permutation))?;
        state.serialize_field(FIELDS[2], data.as_slice())?;
        state.end()
    }
}

// Adapted from https://serde.rs/impl-deserialize.html
impl<'de> Deserialize<'de> for Tensor {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        enum Field {
            Shape,
            Permutation,
            Data,
        }

        impl<'de> Deserialize<'de> for Field {
            fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
            where
                D: serde::Deserializer<'de>,
            {
                struct FieldVisitor;

                impl<'de> Visitor<'de> for FieldVisitor {
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
                            "permutation" => Ok(Field::Permutation),
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
                let shape: Vec<u32> = seq
                    .next_element()?
                    .ok_or_else(|| de::Error::invalid_length(0, &self))?;
                let permutation: Vec<usize> = seq
                    .next_element()?
                    .ok_or_else(|| de::Error::invalid_length(1, &self))?;
                let data: Vec<Complex64> = seq
                    .next_element()?
                    .ok_or_else(|| de::Error::invalid_length(2, &self))?;

                // Create the tensor
                let permutation = raw_to_permutation(&permutation);
                Ok(Tensor {
                    shape: RefCell::new(shape),
                    permutation: RefCell::new(permutation),
                    data: RefCell::new(Rc::new(data)),
                })
            }

            fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
            where
                A: serde::de::MapAccess<'de>,
            {
                let mut shape = None;
                let mut permutation = None;
                let mut data = None;
                while let Some(key) = map.next_key()? {
                    match key {
                        Field::Shape => {
                            if shape.is_some() {
                                return Err(serde::de::Error::duplicate_field("shape"));
                            }
                            shape = Some(map.next_value()?);
                        }
                        Field::Permutation => {
                            if permutation.is_some() {
                                return Err(serde::de::Error::duplicate_field("permutation"));
                            }
                            permutation = Some(map.next_value()?);
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
                let permutation: Vec<usize> =
                    permutation.ok_or_else(|| de::Error::missing_field("permutation"))?;
                let data = data.ok_or_else(|| de::Error::missing_field("data"))?;

                // Create the tensor
                let permutation = raw_to_permutation(&permutation);
                Ok(Tensor {
                    shape: RefCell::new(shape),
                    permutation: RefCell::new(permutation),
                    data: RefCell::new(Rc::new(data)),
                })
            }
        }

        deserializer.deserialize_struct("Tensor", FIELDS, TensorVisitor)
    }
}

#[cfg(test)]
mod tests {
    use num_complex::Complex64;
    use permutation::Permutation;
    use serde::Serialize;

    use crate::{Layout, Tensor};
    use serde_test::{assert_tokens, Token};

    use super::{permutation_to_raw, raw_to_permutation};

    #[derive(Debug)]
    struct TensorEqWrapper(Tensor);

    impl PartialEq for TensorEqWrapper {
        fn eq(&self, other: &Self) -> bool {
            self.0.shape == other.0.shape
                && *self.0.permutation.borrow() == *other.0.permutation.borrow()
                && *self.0.data.borrow() == *other.0.data.borrow()
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
    fn test_permutation_convert() {
        let p1 = Permutation::oneline(vec![2, 0, 1, 3]);
        let p2 = Permutation::oneline(vec![1, 2, 0, 3]).inverse();
        let p3 = &p1 * &p2;
        assert_eq!(raw_to_permutation(&permutation_to_raw(&p1)), p1);
        assert_eq!(raw_to_permutation(&permutation_to_raw(&p2)), p2);
        assert_eq!(raw_to_permutation(&permutation_to_raw(&p3)), p3);
    }

    #[test]
    fn test_serde_scalar() {
        let tensor = Tensor::new_scalar(Complex64::new(1.0, 2.0));

        assert_tokens(
            &TensorEqWrapper(tensor),
            &[
                serde_test::Token::Struct {
                    name: "Tensor",
                    len: 3,
                },
                Token::Str("shape"),
                Token::Seq { len: Some(0) },
                Token::SeqEnd,
                Token::Str("permutation"),
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
        let tensor = Tensor::new_from_flat(&[2, 2, 1], a_data, Some(Layout::RowMajor));

        assert_tokens(
            &TensorEqWrapper(tensor),
            &[
                serde_test::Token::Struct {
                    name: "Tensor",
                    len: 3,
                },
                Token::Str("shape"),
                Token::Seq { len: Some(3) },
                Token::U32(1),
                Token::U32(2),
                Token::U32(2),
                Token::SeqEnd,
                Token::Str("permutation"),
                Token::Seq { len: Some(3) },
                Token::U64(2),
                Token::U64(1),
                Token::U64(0),
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
