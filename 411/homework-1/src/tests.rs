use super::{
    array_set::ArraySet,
    hash_set::HashSet,
    list_set::ListSet,
    tree_set::TreeSet,
    Set,
};

// self-explanatory unit tests

#[test]
fn test_array_set_insert() {
    let mut arr_set = ArraySet::<i32>::new();
    assert!(arr_set.insert(1));
    assert!(arr_set.insert(2));
    assert!(arr_set.insert(3));
    assert!(!arr_set.insert(1));
    assert!(!arr_set.insert(2));
    assert!(!arr_set.insert(3));
}

#[test]
fn test_array_set_remove() {
    let mut arr_set = ArraySet::<i32>::new();
    assert!(!arr_set.remove(1));
    assert!(arr_set.insert(1));
    assert!(arr_set.remove(1));
}

#[test]
fn test_array_set_find() {
    let mut arr_set = ArraySet::<i32>::new();
    assert!(!arr_set.find(1));
    assert!(arr_set.insert(1));
    assert!(arr_set.find(1));
    assert!(arr_set.remove(1));
    assert!(!arr_set.find(1));
}

#[test]
fn test_hash_set_insert() {
    let mut hash_set = HashSet::<i32>::new();
    assert!(hash_set.insert(1));
    assert!(hash_set.insert(2));
    assert!(hash_set.insert(3));
    assert!(!hash_set.insert(1));
    assert!(!hash_set.insert(2));
    assert!(!hash_set.insert(3));
}

#[test]
fn test_hash_set_remove() {
    let mut hash_set = HashSet::<i32>::new();
    assert!(!hash_set.remove(1));
    assert!(hash_set.insert(1));
    assert!(hash_set.remove(1));
}

#[test]
fn test_hash_set_find() {
    let mut hash_set = HashSet::<i32>::new();
    assert!(!hash_set.find(1));
    assert!(hash_set.insert(1));
    assert!(hash_set.find(1));
    assert!(hash_set.remove(1));
    assert!(!hash_set.find(1));
}

#[test]
fn test_list_set_insert() {
    let mut list_set = ListSet::<i32>::new();
    assert!(list_set.insert(1));
    assert!(list_set.insert(2));
    assert!(list_set.insert(3));
    assert!(!list_set.insert(1));
    assert!(!list_set.insert(2));
    assert!(!list_set.insert(3));
}

#[test]
fn test_list_set_remove() {
    let mut list_set = ListSet::<i32>::new();
    assert!(!list_set.remove(1));
    assert!(list_set.insert(1));
    assert!(list_set.remove(1));
}

#[test]
fn test_list_set_find() {
    let mut list_set = ListSet::<i32>::new();
    assert!(!list_set.find(1));
    assert!(list_set.insert(1));
    assert!(list_set.find(1));
    assert!(list_set.remove(1));
    assert!(!list_set.find(1));
}

#[test]
fn test_tree_set_insert() {
    let mut tree_set = TreeSet::<i32>::new();
    assert!(tree_set.insert(1));
    assert!(tree_set.insert(2));
    assert!(tree_set.insert(3));
    assert!(!tree_set.insert(1));
    assert!(!tree_set.insert(2));
    assert!(!tree_set.insert(3));
}

#[test]
fn test_tree_set_remove() {
    let mut tree_set = TreeSet::<i32>::new();
    assert!(!tree_set.remove(1));
    assert!(tree_set.insert(1));
    assert!(tree_set.remove(1));
}

#[test]
fn test_tree_set_find() {
    let mut tree_set = TreeSet::<i32>::new();
    assert!(!tree_set.find(1));
    assert!(tree_set.insert(1));
    assert!(tree_set.find(1));
    assert!(tree_set.remove(1));
    assert!(!tree_set.find(1));
}
