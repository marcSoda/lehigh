use crate::Set;
use std::hash::Hash;
use std::collections::HashSet as HSet;

pub struct HashSet<T> {
    pub set: HSet<T>,
}

impl<T> HashSet<T> {
    pub fn new() -> Self {
        HashSet { set: HSet::new() }
    }
}

impl<T: Eq + Hash> Set<T> for HashSet<T> {
    fn find(&self, val: T) -> bool {
        self.set.contains(&val)
    }

    fn insert(&mut self, val: T) -> bool {
        self.set.insert(val)
    }

    fn remove(&mut self, val: T) -> bool {
        self.set.remove(&val)
    }
}
