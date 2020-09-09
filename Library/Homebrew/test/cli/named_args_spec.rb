# frozen_string_literal: true

require "cli/named_args"

describe Homebrew::CLI::NamedArgs do
  let(:foo) do
    formula "foo" do
      url "https://brew.sh"
      version "1.0"
    end
  end

  let(:bar) do
    formula "bar" do
      url "https://brew.sh"
      version "1.0"
    end
  end

  let(:baz) do
    Cask::CaskLoader.load(+<<~RUBY)
      cask "baz" do
        version "1.0"
      end
    RUBY
  end

  describe "#to_formulae" do
    it "returns formulae" do
      stub_formula_loader foo, call_original: true
      stub_formula_loader bar

      expect(described_class.new("foo", "bar").to_formulae).to eq [foo, bar]
    end

    it "raises an error when a Formula is unavailable" do
      expect { described_class.new("mxcl").to_formulae }.to raise_error FormulaUnavailableError
    end

    it "returns an empty array when there are no Formulae" do
      expect(described_class.new.to_formulae).to be_empty
    end
  end

  describe "#to_formulae_and_casks" do
    it "returns formulae and casks" do
      stub_formula_loader foo, call_original: true
      stub_cask_loader baz, call_original: true

      expect(described_class.new("foo", "baz").to_formulae_and_casks).to eq [foo, baz]
    end
  end

  describe "#to_resolved_formulae" do
    it "returns resolved formulae" do
      allow(Formulary).to receive(:resolve).and_return(foo, bar)

      expect(described_class.new("foo", "bar").to_resolved_formulae).to eq [foo, bar]
    end
  end

  describe "#to_resolved_formulae_to_casks" do
    it "returns resolved formulae, as well as casks" do
      allow(Formulary).to receive(:resolve).and_call_original
      allow(Formulary).to receive(:resolve).with("foo", any_args).and_return foo
      stub_cask_loader baz, call_original: true

      resolved_formulae, casks = described_class.new("foo", "baz").to_resolved_formulae_to_casks

      expect(resolved_formulae).to eq [foo]
      expect(casks).to eq [baz]
    end
  end

  describe "#to_casks" do
    it "returns casks" do
      stub_cask_loader baz

      expect(described_class.new("baz").to_casks).to eq [baz]
    end
  end

  describe "#to_kegs" do
    before do
      (HOMEBREW_CELLAR/"foo/1.0").mkpath
      (HOMEBREW_CELLAR/"bar/1.0").mkpath
    end

    it "resolves kegs with #resolve_kegs" do
      expect(described_class.new("foo", "bar").to_kegs.map(&:name)).to eq ["foo", "bar"]
    end

    it "when there are no matching kegs returns an array of Kegs" do
      expect(described_class.new.to_kegs).to be_empty
    end
  end

  describe "#to_kegs_to_casks" do
    before do
      (HOMEBREW_CELLAR/"foo/1.0").mkpath
    end

    it "returns kegs, as well as casks" do
      stub_cask_loader baz, call_original: true

      kegs, casks = described_class.new("foo", "baz").to_kegs_to_casks

      expect(kegs.map(&:name)).to eq ["foo"]
      expect(casks).to eq [baz]
    end
  end

  describe "#homebrew_tap_cask_names" do
    it "returns an array of casks from homebrew-cask" do
      expect(described_class.new("foo", "homebrew/cask/local-caffeine").homebrew_tap_cask_names)
        .to eq ["homebrew/cask/local-caffeine"]
    end

    it "returns an empty array when there are no matching casks" do
      expect(described_class.new("foo").homebrew_tap_cask_names).to be_empty
    end
  end

  describe "#to_paths" do
    let(:existing_path) { mktmpdir }
    let(:formula_path) { Pathname("/path/to/foo.rb") }
    let(:cask_path) { Pathname("/path/to/baz.rb") }

    before do
      allow(formula_path).to receive(:exist?).and_return(true)
      allow(cask_path).to receive(:exist?).and_return(true)

      allow(Formulary).to receive(:path).and_call_original
      allow(Cask::CaskLoader).to receive(:path).and_call_original
    end

    it "returns taps, cask formula and existing paths" do
      expect(Formulary).to receive(:path).with("foo").and_return(formula_path)
      expect(Cask::CaskLoader).to receive(:path).with("baz").and_return(cask_path)

      expect(described_class.new("homebrew/core", "foo", "baz", existing_path.to_s).to_paths)
        .to eq [Tap.fetch("homebrew/core").path, formula_path, cask_path, existing_path]
    end

    it "returns both cask and formula paths if they exist" do
      expect(Formulary).to receive(:path).with("foo").and_return(formula_path)
      expect(Cask::CaskLoader).to receive(:path).with("baz").and_return(cask_path)

      expect(described_class.new("foo", "baz").to_paths).to eq [formula_path, cask_path]
    end

    it "returns only formulae when `only: :formulae` is specified" do
      expect(Formulary).to receive(:path).with("foo").and_return(formula_path)

      expect(described_class.new("foo", "baz").to_paths(only: :formulae)).to eq [formula_path, Formulary.path("baz")]
    end

    it "returns only casks when `only: :casks` is specified" do
      expect(Cask::CaskLoader).to receive(:path).with("foo").and_return(cask_path)

      expect(described_class.new("foo", "baz").to_paths(only: :casks)).to eq [cask_path, Cask::CaskLoader.path("baz")]
    end
  end
end
