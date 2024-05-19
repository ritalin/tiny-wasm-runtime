#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct FuncType {
    pub params: Vec<ValueType>,
    pub returns: Vec<ValueType>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ValueType {
    I32, // 0x7F
    I64, // 0x7E
}

impl From<u8> for ValueType {
    fn from(value: u8) -> Self {
        match value {
            0x7F => ValueType::I32,
            0x7E => ValueType::I64,
            _ => unreachable!(),
        }
    }
}