use crate::Set;

pub struct ArraySet<T> {
    pub set: Vec<T>,
}

impl<T: PartialEq + Clone> ArraySet<T> {
    pub fn new() -> Self {
        ArraySet { set: Vec::new() }
    }
}

impl<T: PartialEq + Clone> Set<T> for ArraySet<T> {
    fn find(&self, val: T) -> bool {
        self.set.contains(&val)
    }

    fn insert(&mut self, val: T) -> bool {
        match self.find(val.clone()) {
            true => false,
            false => {
                self.set.push(val);
                true
            }
        }
    }

    fn remove(&mut self, val: T) -> bool {
        if let Some(index) = self.set.iter().position(|x| *x == val) {
            self.set.remove(index);
            true
        } else {
            false
        }
    }
}
