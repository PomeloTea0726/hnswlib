/* SIFT1B gt */
void load_data_ivecs(const char* filename, std::vector<std::vector<unsigned>>& data, unsigned num) { // load given num of vecs
  std::ifstream in(filename, std::ios::binary);
  if (!in.is_open()) {
    std::cout << "open file error" << std::endl;
    exit(-1);
  }
  int t;
  in.read((char *) &t, 4);
  data.resize(num, std::vector<unsigned>(t));
  in.seekg(0, std::ios::beg);
  for (unsigned i = 0; i < num; i++) {
      in.read((char *) &t, 4);
      in.read((char *) data[i].data() , t * 4);
  }
  in.close();
}

/* SIFT1B base | query */
void load_data_bvecs(const char* filename, float*& data,
               unsigned& dim, unsigned num, unsigned N_offset) { // load given num of vecs
  std::ifstream in(filename, std::ios::binary);
  if (!in.is_open()) {
    std::cout << "open file error" << std::endl;
    exit(-1);
  }
  in.read((char*)&dim, 4);
  data = new float[(size_t)num * (size_t)dim];

  // skip N_offset vecs
  uint64_t offset_bytes = (uint64_t)(dim + 4) * N_offset;
  in.seekg(offset_bytes, std::ios::beg);
  
  for (size_t i = 0; i < num; i++) {
    in.seekg(4, std::ios::cur);
    for (size_t j = 0; j < dim; j++) {
      unsigned char temp;
      in.read((char*)&temp, 1);
      data[i * dim + j] = (float)temp;
    }
    // in.read((char*)(data + i * dim), dim);
  }
  in.close();
}

/* SBERT base | query */
void load_data_SBERT(const char* filename, float*& data, unsigned num) { // load given num of vecs
  std::ifstream in(filename, std::ios::binary);
  if (!in.is_open()) {
    std::cout << "open file error" << std::endl;
    exit(-1);
  }
  data = new float[(size_t)num * 384];

  for (size_t i = 0; i < num; i++) {
    in.read((char*)(data + i * 384), 384 * sizeof(float));
  }
  in.close();
}

/* Deep | SBERT gt */
void load_data_deep_ibin(const char* filename, std::vector<std::vector<unsigned>>& data, unsigned num) { // load given num of vecs
  std::ifstream in(filename, std::ios::binary);
  if (!in.is_open()) {
    std::cout << "open file error" << std::endl;
    exit(-1);
  }
  in.seekg(4, std::ios::cur);
  int dim;
  in.read((char *) &dim, 4);
  data.resize(num, std::vector<unsigned>(dim));

  for (unsigned i = 0; i < num; i++) {
    in.read((char *) data[i].data(), dim * 4);
  }
  in.close();
}

/* Deep base | query */
void load_data_deep_fbin(const char* filename, float*& data, unsigned num) { // load given num of vecs
  std::ifstream in(filename, std::ios::binary);
  if (!in.is_open()) {
    std::cout << "open file error" << std::endl;
    exit(-1);
  }
  in.seekg(4, std::ios::cur);
  int dim;
  in.read((char *) &dim, 4);
  data = new float[(size_t)num * (size_t)dim];

  for (size_t i = 0; i < num; i++) {
    in.read((char*)(data + i * dim), dim * sizeof(float));
  }
  in.close();
}

/* SPACEV base | query */
void load_data_spacev(const char* filename, float*& data, unsigned num) { // load given num of vecs
  std::ifstream in(filename, std::ios::binary);
  if (!in.is_open()) {
    std::cout << "open file error" << std::endl;
    exit(-1);
  }
  in.seekg(4, std::ios::cur);
  int dim;
  in.read((char *) &dim, 4);
  data = new float[(size_t)num * (size_t)dim];

  int8_t temp;
  for (size_t i = 0; i < num; i++) {
    for (int j = 0; j < dim; j++) {
      in.read((char*)&temp, sizeof(int8_t));
      data[i * dim + j] = (float)temp;
    }
  }

  in.close();
}