pub const HEADER_LEN: usize = 3; // length of each header (PUT, GET, DEL)
pub const STATUS_LEN: usize = 5; // length of each status (padded)

// status returned in Resp for each server response
#[derive(Debug, PartialEq, Clone)]
pub enum Status {
    OK,
    ERR,
    UNKNOWN,
}

// returned by each server response
#[derive(Debug, PartialEq, Clone)]
pub struct Resp {
    pub status: Status,
    pub val: Option<Vec<u8>>,
}
