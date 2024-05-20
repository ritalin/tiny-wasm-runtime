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

#[derive(Debug, PartialEq, Eq)]
pub enum ExportDesc {
    Func(u32),
}

#[derive(Debug, PartialEq, Eq)]
pub struct Export {
    pub name: String,
    pub desc: ExportDesc,
}

#[derive(Debug, PartialEq, Eq)]
pub enum ImportDesc {
    Func(u32),
}

#[derive(Debug, PartialEq, Eq)]
pub struct Import {
    pub mod_name: String,
    pub field_name: String,
    pub desc: ImportDesc,
}

#[derive(Debug, PartialEq, Eq)]
pub struct Memory {
    pub initial: u32,
    pub limit: Option<u32>,
}

#[derive(Debug, PartialEq, Eq)]
pub struct Data {
    pub page: u32,
    pub offset: u32,
    pub bytes: Vec<u8>,
}