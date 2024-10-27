#!/bin/bash

PS1='\[\033[01;32m\]\u@\h\[\033[00m\]:\[\033[01;34m\]\w\[\033[00m\]\$ '

#Add ls colors
export LS_COLORS='di=34:ln=35:so=32:pi=33:ex=31:bd=34;46:cd=34;43:su=30;41:sg=30;46:tw=30;42:ow=30;43'

# colored GCC warnings and errors
export GCC_COLORS='error=01;31:warning=01;35:note=01;36:caret=01;32:locus=01:quote=01'

# enable programmable completion features
if ! shopt -oq posix; then
  if [ -f /usr/share/bash-completion/bash_completion ]; then
    . /usr/share/bash-completion/bash_completion
  elif [ -f /etc/bash_completion ]; then
    . /etc/bash_completion
  fi
fi

#set editor
export EDITOR=vim
export CONAN_USER_HOME=/root/snapkv/conan_home


#system
alias x='startx'
alias c='clear'
alias peg='ps -aux | grep $1'
alias grep='grep --color=auto'
alias ls='ls -la --color=always'
alias rm='rm -v'
alias pg='ping google.com'
#git
alias gp='git push'
alias gs='git status -s'
alias gst='git status'
alias ga='git add'
alias gd='git diff'
alias gck='git checkout'
alias gb='git branch'
alias gl='git log'
gc() {
    git commit -m "$*"
}

