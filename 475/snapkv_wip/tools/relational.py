from jinja2 import Environment, FileSystemLoader
import copy
import sys, os

class Column:
    
    def __init__(self, name : str, ctype : str):
        self.name = name
        self.ctype = ctype

    def __str__(self):
        return self.name + " " + self.ctype

    def __copy__(self):
        return type(self)(self.name, self.ctype)

    def __deepcopy__(self, memo):
        id_self = id(self)
        _copy = memo.get(id_self)
        if _copy is None:
            _copy = type(self)(copy.deepcopy(self.name, memo), copy.deepcopy(self.ctype, memo))
        return _copy

#ColumnList : TypeAlias = list[Column]

class Table:

    def __init__(self, name : str, columns, primaryKey):
        self.name = name
        self.columns = columns
        self.primaryKey = primaryKey

    def isPrimaryKey(self, name : str):
        for p in self.primaryKey:
            if name == p.name:
                return True
        return False

    def __str__(self):
        return self.name + "\n" + str([c.__str__() for c in self.columns]) + str([str(c) for c in self.primaryKey])

class RelationalModel:

    def __init__(self, tables, namespace):
        self.namespace = namespace
        self.tables = tables
        self.tableDict = dict()
        for t in self.tables:
            self.tableDict[t.name] = t

    def __str__(self):
        s = ""
        for t in self.tables:
            s += str(t) + "\n"
        return s

class KeyValueModel:

    def __init__(self, model : RelationalModel):
        self.tableKey = dict()
        for table in model.tables:
            self.tableKey[table.name] = table.primaryKey

class WhereEqual:
    def __init__(self):
        self.one = None
        self.two = None
        self.TwoisLiteral = False

    def givenLiteral(self):
        return self.TwoisLiteral

class WhereEqualColumns(WhereEqual):
    def __init__(self, table1 : str, name1 : str, table2 : str, name2 : str):
        super().__init__()
        self.one = (table1, name1)
        self.two = (table2, name2)

class WhereEqualLiteral(WhereEqual):
    def __init__(self, table : str, name : str):
        super().__init__()
        self.one = (table, name)
        self.TwoisLiteral = True

class QueryOnModel:

    def __init__(self, model : RelationalModel, tablesFrom, conditions):
        self.model = model
        self.conditions = conditions
        self.tablesFrom = dict()
        for t in tablesFrom:
            self.tablesFrom[t] = model.tableDict[t]

    def loopConditions(self):
        potentialConditions = dict()
        for k in self.tablesFrom:
            potentialConditions[k] = (self.tablesFrom[k].primaryKey, list())

        #given the outerloop see if we can get rid of a part of the inner loop
       
        for c in self.conditions:
            if c.givenLiteral():
                if self.tablesFrom[c.one[0]].isPrimaryKey(c.one[1]):
                    print("Given part of primary key for", c.one[0])
                    potentialConditions[c.one[0]][1].append(("Literal", c.one[1]))
            else:
                oneIsPk = self.tablesFrom[c.one[0]].isPrimaryKey(c.one[1]) 
                twoIsPk = self.tablesFrom[c.two[0]].isPrimaryKey(c.two[1])
                if oneIsPk and twoIsPk:
                    potentialConditions[c.one[0]][1].append((c.two[0], c.one[1]))
                    potentialConditions[c.two[0]][1].append((c.one[0], c.two[1]))
            
        #outer = None
        #for k in potentialConditions:
        #    if outer != None:
        #        inner = potentialConditions[k]
        #        matched[0]
        #    outer = potentialConditions[k]

        return potentialConditions



