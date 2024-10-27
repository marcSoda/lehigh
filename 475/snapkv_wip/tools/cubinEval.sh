#Copyright 2022 dePaul Miller
#
#Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
#The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
#
#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
#!/bin/bash

if [[ $# -ne 2 ]]; then
    echo "Usage: $0 <arch> <file>"
    exit 1
fi

x="$(cuobjdump -sass -arch $1 $2)"

if [[ $? -ne 0 ]]; then
    echo $x
    exit 2
fi

functions="$(echo "$x" | grep "Function :" | sed 's/.*Function : \(.*\)/\1/')"

for f in ${functions[@]}; do
    echo "$f" | cu++filt
    regs="$(cuobjdump -sass -arch $1 -fun "$f" $2 | sed -nr 's/.*R([[:digit:]]+).*/\1/p' | sort -n | tail -n1)" 
    echo "$regs registers per thread"
    warpregs=$(($regs * 32))
    echo "Warp needs $warpregs"
done

#x="$(cuobjdump -sass -arch $1 $2)"
