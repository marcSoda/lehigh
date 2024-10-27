#pragma once

#include <string>
#include <vector>

#include "authtableentry.h"

/// The purpose of this file is to allow you to declare helper functions that
/// simplify interacting with persistent storage.

void populate_dat(std::vector<uint8_t> &vec, AuthTableEntry);
bool writef(std::vector<uint8_t>, FILE*);
void push_len_eff(std::vector<uint8_t>&, size_t);
void push_dat_eff(std::vector<uint8_t> &vec, std::vector<uint8_t>);
void push_dat_eff(std::vector<uint8_t> &vec, std::string);
