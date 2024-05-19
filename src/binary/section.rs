use num_derive::FromPrimitive;

#[derive(Debug, PartialEq, Eq, FromPrimitive)]
pub enum SectionCode {
    Type = 0x01,
    Function = 0x03,
    Code = 0x0a,
}

