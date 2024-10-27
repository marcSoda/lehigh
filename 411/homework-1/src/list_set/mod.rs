use crate::Set;
use std::collections::LinkedList;

pub struct ListSet<T> {
    set: LinkedList<T>,
}

impl<T> ListSet<T> {
    pub fn new() -> Self {
        ListSet { set: LinkedList::new() }
    }
}

impl<T: PartialEq + Copy> Set<T> for ListSet<T> {
    fn find(&self, val: T) -> bool {
        self.set.contains(&val)
    }

    fn insert(&mut self, val: T) -> bool {
        match self.find(val) {
            true => false,
            false => {
                self.set.push_back(val);
                true
            }
        }
    }

    fn remove(&mut self, val: T) -> bool {
        let mut idx = 0;
        let mut found = false;
        for (i, v) in self.set.iter().enumerate() {
            if *v == val {
                idx = i;
                found = true;
                break;
            }
        }
        if found { self.set.remove(idx); }
        return found;
    }
}
