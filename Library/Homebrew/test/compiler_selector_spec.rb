# frozen_string_literal: true

require "compilers"
require "software_spec"

describe CompilerSelector do
  subject { described_class.new(software_spec, versions, compilers) }

  let(:compilers) { [:clang, :gnu] }
  let(:software_spec) { SoftwareSpec.new }
  let(:cc) { :clang }
  let(:versions) do
    double(
      llvm_build_version:  Version::NULL,
      clang_build_version: Version.create("600"),
    )
  end

  before do
    allow(versions).to receive(:non_apple_gcc_version) do |name|
      case name
      when "gcc-7" then Version.create("7.1")
      when "gcc-6" then Version.create("6.1")
      else Version::NULL
      end
    end
  end

  describe "#compiler" do
    it "defaults to cc" do
      expect(subject.compiler).to eq(cc)
    end

    it "returns clang if it fails with non-Apple gcc" do
      software_spec.fails_with(gcc: "7")
      expect(subject.compiler).to eq(:clang)
    end

    it "still returns gcc-7 if it fails with gcc without a specific version" do
      software_spec.fails_with(:clang)
      expect(subject.compiler).to eq("gcc-7")
    end

    it "returns gcc-6 if gcc formula offers gcc-6" do
      software_spec.fails_with(:clang)
      allow(Formulary).to receive(:factory).with("gcc").and_return(double(version: "6.0"))
      expect(subject.compiler).to eq("gcc-6")
    end

    it "raises an error when gcc or llvm is missing" do
      software_spec.fails_with(:clang)
      software_spec.fails_with(gcc: "7")
      software_spec.fails_with(gcc: "6")

      expect { subject.compiler }.to raise_error(CompilerSelectionError)
    end
  end
end
