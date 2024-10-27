use crate::Set;
use std::collections::BTreeMap;

pub struct TreeSet<T> {
    pub set: BTreeMap<T, ()>,
}

impl<T: Ord> TreeSet<T> {
    pub fn new() -> Self {
        TreeSet { set: BTreeMap::new() }
    }
}

impl<T: Ord> Set<T> for TreeSet<T> {
    fn find(&self, val: T) -> bool {
        self.set.contains_key(&val)
    }

    fn insert(&mut self, val: T) -> bool {
        self.set.insert(val, ()).is_none()
    }

    fn remove(&mut self, val: T) -> bool {
        self.set.remove(&val).is_some()
    }
}
