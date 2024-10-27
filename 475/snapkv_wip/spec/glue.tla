---- MODULE glue ----
EXTENDS Naturals, Sequences, TLC, Integers, FiniteSets
\* GLUE algorithm

CONSTANTS N, TXS

ASSUME TXSAssumption == TXS > 0

ASSUME NAssumption == N > 0

RECURSIVE NTuple(_)
NTuple(n) == IF n = 0 THEN <<>> ELSE Append(NTuple(n - 1), <<>>)

Max(x, y) == IF x > y THEN x ELSE y

ConsistentSnapshot(ts, seen) == (ts = 0 /\ seen = {}) \/ (\A z \in 1..ts : \E y \in seen : y = z) 

(* --algorithm gluealg

variables timestamp = 0, queue = NTuple(TXS)

define
end define;

\* Push from thread tid
macro Push(tid, x) begin
    queue[tid] := Append(queue[tid], x);
end macro;

\* Pop from thread tid set x and found
macro Pop(tid, x, found) begin
    found := Len(queue[tid]) > 0;
    if found then
        x := Head(queue[tid]);
        queue[tid] := Tail(queue[tid]);
    end if;
end macro;

process commit \in 1..TXS
variables localTs = 0, commitIter = 0
begin
    FOREVER:
    while commitIter < N do
        INCREMENT:
            \* Get timestamp
            localTs := timestamp + 1;
            timestamp := timestamp + 1;
        WRITE:
            \* Log commit
            Push(self, localTs);
        INC:
            commitIter := commitIter + 1
    end while;
end process;

process gpu = 0
variables curr = 0, iter = 0, seen = {}, observed = {}, x = 0, 
found = TRUE, tmp = << >>, tmpLen = 0, numtimes = 0, prev = 0
begin
    FOREVER_2:
    while numtimes < N do
        GET_CURRENT:
            \* Get timestamp
            curr := timestamp;
        INIT:
            iter := 1;
            observed := {};
            tmpLen := Len(tmp);           
        GET_PRIOR_READ:
            \* Read through the prior logs found
            while iter < tmpLen do
                if Head(tmp) <= timestamp then
                    observed := observed \union {Head(tmp)};
                    tmp := Tail(tmp);
                else
                    tmp := Append(Tail(tmp), Head(tmp));
                end if;
                iter := iter + 1;
            end while;
        INIT_2:
            iter := 1;
        READ_UNTIL_FOUND:
            \* read until observed curr
            while Cardinality(observed) < (curr - prev) do  
            READ_QUEUES:
                \* Read through the queues        
                while iter < (TXS + 1) do
                    INIT_IN_LOOP:
                        found := TRUE;
                    INNER_LOOP:
                    while found do
                        READ_A_QUEUE:
                            \* Read from thread iter
                            Pop(iter, x, found);
                        IF_FOUND:
                        if found then
                            IF_VALID:
                            \* if we validate then we update our observed state otherwise we save it for later
                            if x <= curr then
                                observed := observed \union {x}; 
                            else
                                tmp := Append(tmp, x);
                            end if;
                        end if;
                    end while;
                    INCREMENT_ITER:
                        iter := iter + 1;
                end while;
            end while;
        FINISHED_READ:
            prev := curr;
            seen := seen \union observed;
            \* Assertion that we have seen all of the state
        CHECK_INVARIANT:
            assert ConsistentSnapshot(curr, seen); 
        INCREMENT_NUM_TIMES:
            numtimes := numtimes + 1;
    end while;
end process;


end algorithm; *)
\* BEGIN TRANSLATION (chksum(pcal) = "fbbfe361" /\ chksum(tla) = "47d52fb2")
VARIABLES timestamp, queue, pc, localTs, commitIter, curr, iter, seen, 
          observed, x, found, tmp, tmpLen, numtimes, prev

vars == << timestamp, queue, pc, localTs, commitIter, curr, iter, seen, 
           observed, x, found, tmp, tmpLen, numtimes, prev >>

ProcSet == (1..TXS) \cup {0}

Init == (* Global variables *)
        /\ timestamp = 0
        /\ queue = NTuple(TXS)
        (* Process commit *)
        /\ localTs = [self \in 1..TXS |-> 0]
        /\ commitIter = [self \in 1..TXS |-> 0]
        (* Process gpu *)
        /\ curr = 0
        /\ iter = 0
        /\ seen = {}
        /\ observed = {}
        /\ x = 0
        /\ found = TRUE
        /\ tmp = << >>
        /\ tmpLen = 0
        /\ numtimes = 0
        /\ prev = 0
        /\ pc = [self \in ProcSet |-> CASE self \in 1..TXS -> "FOREVER"
                                        [] self = 0 -> "FOREVER_2"]

FOREVER(self) == /\ pc[self] = "FOREVER"
                 /\ IF commitIter[self] < N
                       THEN /\ pc' = [pc EXCEPT ![self] = "INCREMENT"]
                       ELSE /\ pc' = [pc EXCEPT ![self] = "Done"]
                 /\ UNCHANGED << timestamp, queue, localTs, commitIter, curr, 
                                 iter, seen, observed, x, found, tmp, tmpLen, 
                                 numtimes, prev >>

INCREMENT(self) == /\ pc[self] = "INCREMENT"
                   /\ localTs' = [localTs EXCEPT ![self] = timestamp + 1]
                   /\ timestamp' = timestamp + 1
                   /\ pc' = [pc EXCEPT ![self] = "WRITE"]
                   /\ UNCHANGED << queue, commitIter, curr, iter, seen, 
                                   observed, x, found, tmp, tmpLen, numtimes, 
                                   prev >>

WRITE(self) == /\ pc[self] = "WRITE"
               /\ queue' = [queue EXCEPT ![self] = Append(queue[self], localTs[self])]
               /\ pc' = [pc EXCEPT ![self] = "INC"]
               /\ UNCHANGED << timestamp, localTs, commitIter, curr, iter, 
                               seen, observed, x, found, tmp, tmpLen, numtimes, 
                               prev >>

INC(self) == /\ pc[self] = "INC"
             /\ commitIter' = [commitIter EXCEPT ![self] = commitIter[self] + 1]
             /\ pc' = [pc EXCEPT ![self] = "FOREVER"]
             /\ UNCHANGED << timestamp, queue, localTs, curr, iter, seen, 
                             observed, x, found, tmp, tmpLen, numtimes, prev >>

commit(self) == FOREVER(self) \/ INCREMENT(self) \/ WRITE(self)
                   \/ INC(self)

FOREVER_2 == /\ pc[0] = "FOREVER_2"
             /\ IF numtimes < N
                   THEN /\ pc' = [pc EXCEPT ![0] = "GET_CURRENT"]
                   ELSE /\ pc' = [pc EXCEPT ![0] = "Done"]
             /\ UNCHANGED << timestamp, queue, localTs, commitIter, curr, iter, 
                             seen, observed, x, found, tmp, tmpLen, numtimes, 
                             prev >>

GET_CURRENT == /\ pc[0] = "GET_CURRENT"
               /\ curr' = timestamp
               /\ pc' = [pc EXCEPT ![0] = "INIT"]
               /\ UNCHANGED << timestamp, queue, localTs, commitIter, iter, 
                               seen, observed, x, found, tmp, tmpLen, numtimes, 
                               prev >>

INIT == /\ pc[0] = "INIT"
        /\ iter' = 1
        /\ observed' = {}
        /\ tmpLen' = Len(tmp)
        /\ pc' = [pc EXCEPT ![0] = "GET_PRIOR_READ"]
        /\ UNCHANGED << timestamp, queue, localTs, commitIter, curr, seen, x, 
                        found, tmp, numtimes, prev >>

GET_PRIOR_READ == /\ pc[0] = "GET_PRIOR_READ"
                  /\ IF iter < tmpLen
                        THEN /\ IF Head(tmp) <= timestamp
                                   THEN /\ observed' = (observed \union {Head(tmp)})
                                        /\ tmp' = Tail(tmp)
                                   ELSE /\ tmp' = Append(Tail(tmp), Head(tmp))
                                        /\ UNCHANGED observed
                             /\ iter' = iter + 1
                             /\ pc' = [pc EXCEPT ![0] = "GET_PRIOR_READ"]
                        ELSE /\ pc' = [pc EXCEPT ![0] = "INIT_2"]
                             /\ UNCHANGED << iter, observed, tmp >>
                  /\ UNCHANGED << timestamp, queue, localTs, commitIter, curr, 
                                  seen, x, found, tmpLen, numtimes, prev >>

INIT_2 == /\ pc[0] = "INIT_2"
          /\ iter' = 1
          /\ pc' = [pc EXCEPT ![0] = "READ_UNTIL_FOUND"]
          /\ UNCHANGED << timestamp, queue, localTs, commitIter, curr, seen, 
                          observed, x, found, tmp, tmpLen, numtimes, prev >>

READ_UNTIL_FOUND == /\ pc[0] = "READ_UNTIL_FOUND"
                    /\ IF Cardinality(observed) < (curr - prev)
                          THEN /\ pc' = [pc EXCEPT ![0] = "READ_QUEUES"]
                          ELSE /\ pc' = [pc EXCEPT ![0] = "FINISHED_READ"]
                    /\ UNCHANGED << timestamp, queue, localTs, commitIter, 
                                    curr, iter, seen, observed, x, found, tmp, 
                                    tmpLen, numtimes, prev >>

READ_QUEUES == /\ pc[0] = "READ_QUEUES"
               /\ IF iter < (TXS + 1)
                     THEN /\ pc' = [pc EXCEPT ![0] = "INIT_IN_LOOP"]
                     ELSE /\ pc' = [pc EXCEPT ![0] = "READ_UNTIL_FOUND"]
               /\ UNCHANGED << timestamp, queue, localTs, commitIter, curr, 
                               iter, seen, observed, x, found, tmp, tmpLen, 
                               numtimes, prev >>

INIT_IN_LOOP == /\ pc[0] = "INIT_IN_LOOP"
                /\ found' = TRUE
                /\ pc' = [pc EXCEPT ![0] = "INNER_LOOP"]
                /\ UNCHANGED << timestamp, queue, localTs, commitIter, curr, 
                                iter, seen, observed, x, tmp, tmpLen, numtimes, 
                                prev >>

INNER_LOOP == /\ pc[0] = "INNER_LOOP"
              /\ IF found
                    THEN /\ pc' = [pc EXCEPT ![0] = "READ_A_QUEUE"]
                    ELSE /\ pc' = [pc EXCEPT ![0] = "INCREMENT_ITER"]
              /\ UNCHANGED << timestamp, queue, localTs, commitIter, curr, 
                              iter, seen, observed, x, found, tmp, tmpLen, 
                              numtimes, prev >>

READ_A_QUEUE == /\ pc[0] = "READ_A_QUEUE"
                /\ found' = (Len(queue[iter]) > 0)
                /\ IF found'
                      THEN /\ x' = Head(queue[iter])
                           /\ queue' = [queue EXCEPT ![iter] = Tail(queue[iter])]
                      ELSE /\ TRUE
                           /\ UNCHANGED << queue, x >>
                /\ pc' = [pc EXCEPT ![0] = "IF_FOUND"]
                /\ UNCHANGED << timestamp, localTs, commitIter, curr, iter, 
                                seen, observed, tmp, tmpLen, numtimes, prev >>

IF_FOUND == /\ pc[0] = "IF_FOUND"
            /\ IF found
                  THEN /\ pc' = [pc EXCEPT ![0] = "IF_VALID"]
                  ELSE /\ pc' = [pc EXCEPT ![0] = "INNER_LOOP"]
            /\ UNCHANGED << timestamp, queue, localTs, commitIter, curr, iter, 
                            seen, observed, x, found, tmp, tmpLen, numtimes, 
                            prev >>

IF_VALID == /\ pc[0] = "IF_VALID"
            /\ IF x <= curr
                  THEN /\ observed' = (observed \union {x})
                       /\ tmp' = tmp
                  ELSE /\ tmp' = Append(tmp, x)
                       /\ UNCHANGED observed
            /\ pc' = [pc EXCEPT ![0] = "INNER_LOOP"]
            /\ UNCHANGED << timestamp, queue, localTs, commitIter, curr, iter, 
                            seen, x, found, tmpLen, numtimes, prev >>

INCREMENT_ITER == /\ pc[0] = "INCREMENT_ITER"
                  /\ iter' = iter + 1
                  /\ pc' = [pc EXCEPT ![0] = "READ_QUEUES"]
                  /\ UNCHANGED << timestamp, queue, localTs, commitIter, curr, 
                                  seen, observed, x, found, tmp, tmpLen, 
                                  numtimes, prev >>

FINISHED_READ == /\ pc[0] = "FINISHED_READ"
                 /\ prev' = curr
                 /\ seen' = (seen \union observed)
                 /\ pc' = [pc EXCEPT ![0] = "CHECK_INVARIANT"]
                 /\ UNCHANGED << timestamp, queue, localTs, commitIter, curr, 
                                 iter, observed, x, found, tmp, tmpLen, 
                                 numtimes >>

CHECK_INVARIANT == /\ pc[0] = "CHECK_INVARIANT"
                   /\ Assert(ConsistentSnapshot(curr, seen), 
                             "Failure of assertion at line 115, column 13.")
                   /\ pc' = [pc EXCEPT ![0] = "INCREMENT_NUM_TIMES"]
                   /\ UNCHANGED << timestamp, queue, localTs, commitIter, curr, 
                                   iter, seen, observed, x, found, tmp, tmpLen, 
                                   numtimes, prev >>

INCREMENT_NUM_TIMES == /\ pc[0] = "INCREMENT_NUM_TIMES"
                       /\ numtimes' = numtimes + 1
                       /\ pc' = [pc EXCEPT ![0] = "FOREVER_2"]
                       /\ UNCHANGED << timestamp, queue, localTs, commitIter, 
                                       curr, iter, seen, observed, x, found, 
                                       tmp, tmpLen, prev >>

gpu == FOREVER_2 \/ GET_CURRENT \/ INIT \/ GET_PRIOR_READ \/ INIT_2
          \/ READ_UNTIL_FOUND \/ READ_QUEUES \/ INIT_IN_LOOP \/ INNER_LOOP
          \/ READ_A_QUEUE \/ IF_FOUND \/ IF_VALID \/ INCREMENT_ITER
          \/ FINISHED_READ \/ CHECK_INVARIANT \/ INCREMENT_NUM_TIMES

(* Allow infinite stuttering to prevent deadlock on termination. *)
Terminating == /\ \A self \in ProcSet: pc[self] = "Done"
               /\ UNCHANGED vars

Next == gpu
           \/ (\E self \in 1..TXS: commit(self))
           \/ Terminating

Spec == Init /\ [][Next]_vars

Termination == <>(\A self \in ProcSet: pc[self] = "Done")

\* END TRANSLATION 
InductiveInvariant == <>ConsistentSnapshot(prev, seen)

THEOREM InitialCondition == Init => ConsistentSnapshot(prev, seen)
    BY NAssumption, TXSAssumption DEF Init, InductiveInvariant, ConsistentSnapshot

=====
