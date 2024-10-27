#![feature(linked_list_remove)]

pub mod array_set;
pub mod hash_set;
pub mod list_set;
pub mod tree_set;
#[cfg(test)]
mod tests;

// Each Set implements the Set trait. Reduces code duplication.
pub trait Set<T> {
    // return true if value exists in set. false otherwise
    fn find(&self, val: T) -> bool;
    // insert value into the set. true on success, false otherwise
    fn insert(&mut self, val: T) -> bool;
    // remove value from set. true on success, false otherwise
    fn remove(&mut self, val: T) -> bool;
}

// To specify the type of set
pub enum SetType {
    ArraySet,
    HashSet,
    ListSet,
    TreeSet,
}
