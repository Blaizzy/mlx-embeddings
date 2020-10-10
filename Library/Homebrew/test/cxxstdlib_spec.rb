# typed: false
# frozen_string_literal: true

require "formula"
require "cxxstdlib"

describe CxxStdlib do
  let(:clang) { described_class.create(:libstdcxx, :clang) }
  let(:gcc6) { described_class.create(:libstdcxx, "gcc-6") }
  let(:gcc7) { described_class.create(:libstdcxx, "gcc-7") }
  let(:lcxx) { described_class.create(:libcxx, :clang) }
  let(:purec) { described_class.create(nil, :clang) }

  describe "#compatible_with?" do
    specify "compatibility with itself" do
      expect(gcc7).to be_compatible_with(gcc7)
      expect(clang).to be_compatible_with(clang)
    end

    specify "Apple/GNU libstdcxx incompatibility" do
      expect(clang).not_to be_compatible_with(gcc7)
      expect(gcc7).not_to be_compatible_with(clang)
    end

    specify "GNU cross-version incompatibility" do
      expect(gcc6).not_to be_compatible_with(gcc7)
      expect(gcc7).not_to be_compatible_with(gcc6)
    end

    specify "libstdcxx and libcxx incompatibility" do
      expect(clang).not_to be_compatible_with(lcxx)
      expect(lcxx).not_to be_compatible_with(clang)
    end

    specify "compatibility for non-cxx software" do
      expect(purec).to be_compatible_with(clang)
      expect(clang).to be_compatible_with(purec)
      expect(purec).to be_compatible_with(purec)
      expect(purec).to be_compatible_with(gcc7)
      expect(gcc7).to be_compatible_with(purec)
    end
  end

  describe "#apple_compiler?" do
    it "returns true for Apple compilers" do
      expect(clang).to be_an_apple_compiler
    end

    it "returns false for non-Apple compilers" do
      expect(gcc7).not_to be_an_apple_compiler
    end
  end

  describe "#type_string" do
    specify "formatting" do
      expect(clang.type_string).to eq("libstdc++")
      expect(lcxx.type_string).to eq("libc++")
    end
  end
end
