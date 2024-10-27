#include <cstdio>
#include <string>
#include <unistd.h>
#include <vector>

#include "persist.h"
#include "authtableentry.h"
#include "format.h"


using namespace std;

void populate_dat(std::vector<uint8_t> &dat, AuthTableEntry u) {
  dat.clear();
  dat.insert(dat.begin(), AUTHENTRY.begin(), AUTHENTRY.end());
  push_len_eff(dat, u.username.size());
  push_len_eff(dat, u.salt.size());
  push_len_eff(dat, u.pass_hash.size());
  push_len_eff(dat, u.content.size());
  push_dat_eff(dat, u.username);
  push_dat_eff(dat, u.salt);
  push_dat_eff(dat, u.pass_hash);
  push_dat_eff(dat, u.content);
  int pad = 8 - ((u.username.length() + u.salt.size() + u.pass_hash.size() + u.content.size()) % 8);
  dat.resize(dat.size() + pad, 0);
}

bool writef(vector<uint8_t> dat, FILE *file) {
  if (fwrite(dat.data(), sizeof(char), dat.size(), file) != dat.size()) return false;
  if (!fflush(file) == 0) return false;
  if (!fsync(fileno(file)) == 0) return false;
  return true;
}

void push_len_eff(std::vector<uint8_t> &vec, size_t len) {
  vec.insert(vec.end(), (uint8_t*)(&len), (uint8_t*)(&len) + sizeof(len));
}

void push_dat_eff(std::vector<uint8_t> &vec, std::vector<uint8_t> dat) {
  vec.insert(vec.end(), dat.begin(), dat.end());
}

void push_dat_eff(std::vector<uint8_t> &vec, string dat) {
  vec.insert(vec.end(), dat.begin(), dat.end());
}
