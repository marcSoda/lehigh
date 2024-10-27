#include <tpccTx.h>
#include <vector>

#pragma once

struct Q2Output {
    int suppkey;
    int su_name;
    int n_name;
    int i_id;
    int i_name;
    int su_address;
    int su_phone;
    int su_comment;
};

template<typename T>
void runq2(T& htap, std::string q2Name) {
    using namespace tpcc;

    std::vector<Key> q2Keys;
    Key k;

    // query item, stock, supplier, nation, region

    // first we need to group by s_i_id and check that the 
    // mod((s_w_id*s_i_id),10000)=su_suppkey the supplier nation key
    // and the nation key are the same
    // the nation region key and the region key are the same
    // then we check that the region has the letters 'Europ'
    // From this we get the stocks item id and the minimum quantity
    
    // Then we take the item id and the minimum quantity and check that
    // for a given item with the supply key condition
    // where the nation keys and region keys are the same and the item data has the last character b
    // and the region name is like 'Europ'
    // and the item id is the item with the item id returned and the quantity is the lowest quantity
    

    htap.query(q2Keys, q2Name, nullptr);
}
