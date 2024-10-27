from relational import *

def getModel():
    ID = Column("id", "uint64_t")
    USER = Column("user", "uint64_t")
    LAT = Column("latitude", "float")
    LONG = Column("longitude", "float")
    
    userColumns = [USER, ID]
    userPk = [USER]
    user = Table("USERTABLE", userColumns, userPk)

    txColumns = [USER, ID, LONG, LAT]
    txPk = [USER, ID]
    tx = Table("TRANSACTION", txColumns, txPk)

    model = RelationalModel([user, tx], 'ml')

    return model

