#include <SiKV.h>
#include <tpcc.h>

using namespace tpcc;

template<int LSLAB_SIZE>
void considerHash() {
    Key k;

    std::cout << "Considering size " << LSLAB_SIZE << std::endl;
    
    sikv::OrderPreservingHash<Key, LSLAB_SIZE> hash{};

    unsigned start;
    unsigned end;

    std::tie(start, end) = hash.getRange(WAREHOUSE); 

    std::cout << "Warehouse range is " <<  start << " to " << end << std::endl;

    //-------------------------

    std::tie(start, end) = hash.getRange(DISTRICT); 

    std::cout << "District range is " <<  start << " to " << end << std::endl;

      
//    //-------------------------
//    
//    PrimaryKey<NEWORDER> neworderPk;
//    
//    neworderPk.did = 1;
//    neworderPk.oid = 1;
//    neworderPk.wid = 1;
//    
//    //-------------------------
//    PrimaryKey<CUSTOMER> customerPk;
//    
//    customerPk.did = 1;
//    customerPk.id = 1;
//    customerPk.wid = 1;
//    
//    //-------------------------
//    PrimaryKey<HISTORY> historyPk;
//   
//    historyPk.cdid = 1;
//    historyPk.cid = 1;
//    historyPk.cwid = 1;
//    historyPk.did = 1;
//    historyPk.wid = 1; 
//    
//    //-------------------------
//    PrimaryKey<ORDER> orderPk;
//   
//    orderPk.did = 1;
//    orderPk.id = 1;
//    orderPk.wid = 1; 
//    
//    //-------------------------
//    PrimaryKey<ORDERLINE> orderlinePk;
//   
//    orderlinePk.did = 1; 
//    orderlinePk.number = 1; 
//    orderlinePk.oid = 1; 
//    orderlinePk.wid = 1; 
//    
//    //-------------------------
//    PrimaryKey<ITEM> itemPk;
//  
//    itemPk.id = 1;  
//    
//    //-------------------------
//    PrimaryKey<STOCK> stockPk;
//   
//    stockPk.iid = 1;
//    stockPk.wid = 1; 
//    
//    //-------------------------
//    PrimaryKey<SUPPLIER> supplierPk;
//   
//    supplierPk.id = 1; 
//    
//    //-------------------------
//    PrimaryKey<NATION> nationPk;
//    
//    nationPk.id = 1;
//    
//    //-------------------------
//    PrimaryKey<REGION> regionPk;
//
//    regionPk.id = 1;


}

int main(int, char**) {

    considerHash<1 << 10>();
    considerHash<1 << 12>();
    considerHash<1 << 14>();
    considerHash<1 << 20>();

    return 0;
}
