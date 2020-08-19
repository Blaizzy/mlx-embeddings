# frozen_string_literal: true

require "cli/named_args"

describe Homebrew::CLI::NamedArgs do
  let(:foo) do
    formula "foo" do
      url "https://brew.sh"
      version "1.0"
    end
  end

  let(:foo_keg) do
    path = (HOMEBREW_CELLAR/"foo/1.0").resolved_path
    mkdir_p path
    Keg.new(path)
  end

  let(:bar) do
    formula "bar" do
      url "https://brew.sh"
      version "1.0"
    end
  end

  let(:bar_keg) do
    path = (HOMEBREW_CELLAR/"bar/1.0").resolved_path
    mkdir_p path
    Keg.new(path)
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
    it "returns kegs" do
      named_args = described_class.new("foo", "bar")
      allow(named_args).to receive(:resolve_keg).with("foo").and_return foo_keg
      allow(named_args).to receive(:resolve_keg).with("bar").and_return bar_keg

      expect(named_args.to_kegs).to eq [foo_keg, bar_keg]
    end
  end

  describe "#to_kegs_to_casks" do
    it "returns kegs, as well as casks" do
      named_args = described_class.new("foo", "baz")
      allow(named_args).to receive(:resolve_keg).and_call_original
      allow(named_args).to receive(:resolve_keg).with("foo").and_return foo_keg
      stub_cask_loader baz, call_original: true

      kegs, casks = named_args.to_kegs_to_casks

      expect(kegs).to eq [foo_keg]
      expect(casks).to eq [baz]
    end
  end
end
