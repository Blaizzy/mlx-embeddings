# typed: false
# frozen_string_literal: true

require "compilers"
require "software_spec"

describe CompilerSelector do
  subject(:selector) { described_class.new(software_spec, versions, compilers) }

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
      when "gcc-5" then Version.create("5.1")
      else Version::NULL
      end
    end
  end

  describe "#compiler" do
    it "defaults to cc" do
      expect(selector.compiler).to eq(cc)
    end

    it "returns clang if it fails with non-Apple gcc" do
      software_spec.fails_with(gcc: "7")
      expect(selector.compiler).to eq(:clang)
    end

    it "still returns gcc-7 if it fails with gcc without a specific version" do
      software_spec.fails_with(:clang)
      expect(selector.compiler).to eq("gcc-7")
    end

    it "returns gcc-6 if gcc formula offers gcc-6 on mac", :needs_macos do
      software_spec.fails_with(:clang)
      allow(Formulary).to receive(:factory).with("gcc").and_return(double(version: "6.0"))
      expect(selector.compiler).to eq("gcc-6")
    end

    it "returns gcc-5 if gcc formula offers gcc-5 on linux", :needs_linux do
      software_spec.fails_with(:clang)
      allow(Formulary).to receive(:factory).with("gcc@5").and_return(double(version: "5.0"))
      expect(selector.compiler).to eq("gcc-5")
    end

    it "returns gcc-6 if gcc formula offers gcc-6 and fails with gcc-5 and gcc-7 on linux", :needs_linux do
      software_spec.fails_with(:clang)
      software_spec.fails_with(gcc: "5")
      software_spec.fails_with(gcc: "7")
      allow(Formulary).to receive(:factory).with("gcc@5").and_return(double(version: "5.0"))
      expect(selector.compiler).to eq("gcc-6")
    end

    it "raises an error when gcc or llvm is missing" do
      software_spec.fails_with(:clang)
      software_spec.fails_with(gcc: "7")
      software_spec.fails_with(gcc: "6")
      software_spec.fails_with(gcc: "5")

      expect { selector.compiler }.to raise_error(CompilerSelectionError)
    end
  end
end
