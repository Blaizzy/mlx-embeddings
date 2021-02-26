# typed: false
# frozen_string_literal: true

require "exceptions"

describe "Exception" do
  describe MultipleVersionsInstalledError do
    subject {
      described_class.new <<~EOS
        foo has multiple installed versions
        Run `brew uninstall --force foo` to remove all versions.
      EOS
    }

    its(:to_s) {
      is_expected.to eq <<~EOS
        foo has multiple installed versions
        Run `brew uninstall --force foo` to remove all versions.
      EOS
    }
  end

  describe NoSuchKegError do
    subject { described_class.new("foo") }

    its(:to_s) { is_expected.to eq("No such keg: #{HOMEBREW_CELLAR}/foo") }
  end

  describe FormulaValidationError do
    subject(:error) { described_class.new("foo", "sha257", "magic") }

    its(:to_s) {
      expect(error.to_s).to eq(%q(invalid attribute for formula 'foo': sha257 ("magic")))
    }
  end

  describe FormulaUnavailableError do
    subject(:error) { described_class.new("foo") }

    describe "#dependent_s" do
      it "returns nil if there is no dependent" do
        expect(error.dependent_s).to be nil
      end

      it "returns nil if it depended on by itself" do
        error.dependent = "foo"
        expect(error.dependent_s).to be nil
      end

      it "returns a string if there is a dependent" do
        error.dependent = "foobar"
        expect(error.dependent_s).to eq(" (dependency of foobar)")
      end
    end

    context "without a dependent" do
      its(:to_s) { is_expected.to eq('No available formula with the name "foo".') }
    end

    context "with a dependent" do
      before do
        error.dependent = "foobar"
      end

      its(:to_s) {
        expect(error.to_s).to eq('No available formula with the name "foo" (dependency of foobar).')
      }
    end
  end

  describe TapFormulaUnavailableError do
    subject { described_class.new(tap, "foo") }

    let(:tap) { double(Tap, user: "u", repo: "r", to_s: "u/r", installed?: false) }

    its(:to_s) { is_expected.to match(%r{Please tap it and then try again: brew tap u/r}) }
  end

  describe FormulaClassUnavailableError do
    subject(:error) { described_class.new("foo", "foo.rb", "Foo", list) }

    let(:mod) do
      Module.new do
        class Bar < Requirement; end

        class Baz < Formula; end
      end
    end

    context "when there are no classes" do
      let(:list) { [] }

      its(:to_s) {
        expect(error.to_s).to match(/Expected to find class Foo, but found no classes\./)
      }
    end

    context "when the class is not derived from Formula" do
      let(:list) { [mod.const_get(:Bar)] }

      its(:to_s) {
        expect(error.to_s).to match(/Expected to find class Foo, but only found: Bar \(not derived from Formula!\)\./)
      }
    end

    context "when the class is derived from Formula" do
      let(:list) { [mod.const_get(:Baz)] }

      its(:to_s) { is_expected.to match(/Expected to find class Foo, but only found: Baz\./) }
    end
  end

  describe FormulaUnreadableError do
    subject { described_class.new("foo", formula_error) }

    let(:formula_error) { LoadError.new("bar") }

    its(:to_s) { is_expected.to eq("foo: bar") }
  end

  describe TapUnavailableError do
    subject { described_class.new("foo") }

    its(:to_s) { is_expected.to eq("No available tap foo.\n") }
  end

  describe TapAlreadyTappedError do
    subject { described_class.new("foo") }

    its(:to_s) { is_expected.to eq("Tap foo already tapped.\n") }
  end

  describe BuildError do
    subject { described_class.new(formula, "badprg", %w[arg1 arg2], {}) }

    let(:formula) { double(Formula, name: "foo") }

    its(:to_s) { is_expected.to eq("Failed executing: badprg arg1 arg2") }
  end

  describe OperationInProgressError do
    subject { described_class.new("foo") }

    its(:to_s) { is_expected.to match(/Operation already in progress for foo/) }
  end

  describe FormulaInstallationAlreadyAttemptedError do
    subject { described_class.new(formula) }

    let(:formula) { double(Formula, full_name: "foo/bar") }

    its(:to_s) { is_expected.to eq("Formula installation already attempted: foo/bar") }
  end

  describe FormulaConflictError do
    subject { described_class.new(formula, [conflict]) }

    let(:formula) { double(Formula, full_name: "foo/qux") }
    let(:conflict) { double(name: "bar", reason: "I decided to") }

    its(:to_s) { is_expected.to match(/Please `brew unlink bar` before continuing\./) }
  end

  describe CompilerSelectionError do
    subject { described_class.new(formula) }

    let(:formula) { double(Formula, full_name: "foo") }

    its(:to_s) { is_expected.to match(/foo cannot be built with any available compilers\./) }
  end

  describe CurlDownloadStrategyError do
    context "when the file does not exist" do
      subject { described_class.new("file:///tmp/foo") }

      its(:to_s) { is_expected.to eq("File does not exist: /tmp/foo") }
    end

    context "when the download failed" do
      subject { described_class.new("https://brew.sh") }

      its(:to_s) { is_expected.to eq("Download failed: https://brew.sh") }
    end
  end

  describe ErrorDuringExecution do
    subject { described_class.new(["badprg", "arg1", "arg2"], status: status) }

    let(:status) { instance_double(Process::Status, exitstatus: 17, termsig: nil) }

    its(:to_s) { is_expected.to eq("Failure while executing; `badprg arg1 arg2` exited with 17.") }
  end

  describe ChecksumMismatchError do
    subject { described_class.new("/file.tar.gz", hash1, hash2) }

    let(:hash1) { double(hash_type: "sha256", to_s: "deadbeef") }
    let(:hash2) { double(hash_type: "sha256", to_s: "deadcafe") }

    its(:to_s) { is_expected.to match(/SHA256 mismatch/) }
  end

  describe ResourceMissingError do
    subject { described_class.new(formula, resource) }

    let(:formula) { double(Formula, full_name: "bar") }
    let(:resource) { double(inspect: "<resource foo>") }

    its(:to_s) { is_expected.to eq("bar does not define resource <resource foo>") }
  end

  describe DuplicateResourceError do
    subject { described_class.new(resource) }

    let(:resource) { double(inspect: "<resource foo>") }

    its(:to_s) { is_expected.to eq("Resource <resource foo> is defined more than once") }
  end

  describe BottleFormulaUnavailableError do
    subject { described_class.new("/foo.bottle.tar.gz", "foo/1.0/.brew/foo.rb") }

    let(:formula) { double(Formula, full_name: "foo") }

    its(:to_s) { is_expected.to match(/This bottle does not contain the formula file/) }
  end

  describe BuildFlagsError do
    subject { described_class.new(["-s"]) }

    its(:to_s) { is_expected.to match(/flag:\s+-s\nrequires building tools/) }
  end
end
