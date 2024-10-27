from conans import ConanFile, CMake

class RSlabConan(ConanFile):
    name = "rslab"
    version = "2.2.1"
    author = "dePaul Miller"
    url = "https://github.com/depaulmillz/rslab"
    license = "MIT"
    settings="os", "compiler", "build_type", "arch"
    requires="unifiedmemorygroupallocation/1.1", "culog/0.1"
    build_requires="gtest/1.10.0"
    generators="cmake"
    
    topic = ("data structure", "gpu programming", "hashmap", "hashtable")
    
    exports_sources = "CMakeLists.txt", "cmake/*", "benchmark/*", "include/*", "test/*", "LICENSE", "Doxygen", "README.md"
    options = {"cuda_arch" : "ANY", "cuda_compiler" : "ANY"}

    def configure(self):
        if self.options.cuda_arch == None:
            self.options.cuda_arch = '70;75;80;86'
        if self.options.cuda_compiler == None:
            self.options.cuda_compiler = "nvcc"

    def _configure_cmake(self):
        cmake = CMake(self)
        cmake.definitions["CMAKE_CUDA_ARCHITECTURES"] = str(self.options.cuda_arch)
        cmake.definitions["CMAKE_CUDA_COMPILER"] = str(self.options.cuda_compiler)
        cmake.definitions["USING_CONAN"] = "ON"
        cmake.definitions["CMAKE_EXPORT_COMPILE_COMMANDS"] = "ON"
        cmake.configure()
        return cmake

    def build(self):
        cmake = self._configure_cmake()
        cmake.build()
        #cmake.test()

    def package(self):
        cmake = self._configure_cmake()
        cmake.install()

    def package_id(self):
        self.info.header_only()

    def package_info(self):
        self.cpp_info.names["cmake_find_package"] = "LSlab"
        self.cpp_info.names["cmake_find_package_multi"] = "LSlab"
