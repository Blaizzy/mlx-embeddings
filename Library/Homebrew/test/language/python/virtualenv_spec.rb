# typed: false
# frozen_string_literal: true

require "language/python"
require "resource"

describe Language::Python::Virtualenv::Virtualenv, :needs_python do
  subject(:virtualenv) { described_class.new(formula, dir, "python") }

  let(:dir) { mktmpdir }

  let(:resource) { double("resource", stage: true) }
  let(:formula_bin) { dir/"formula_bin" }
  let(:formula) { double("formula", resource: resource, bin: formula_bin) }

  describe "#create" do
    it "creates a venv" do
      expect(formula).to receive(:system).with("python", "-m", "venv", "--system-site-packages", dir)
      virtualenv.create
    end
  end

  describe "#pip_install" do
    it "accepts a string" do
      expect(formula).to receive(:system)
        .with(dir/"bin/pip", "install", "-v", "--no-deps",
              "--no-binary", ":all:", "--ignore-installed", "foo")
        .and_return(true)
      virtualenv.pip_install "foo"
    end

    it "accepts a multi-line strings" do
      expect(formula).to receive(:system)
        .with(dir/"bin/pip", "install", "-v", "--no-deps",
              "--no-binary", ":all:", "--ignore-installed", "foo", "bar")
        .and_return(true)

      virtualenv.pip_install <<~EOS
        foo
        bar
      EOS
    end

    it "accepts an array" do
      expect(formula).to receive(:system)
        .with(dir/"bin/pip", "install", "-v", "--no-deps",
              "--no-binary", ":all:", "--ignore-installed", "foo")
        .and_return(true)

      expect(formula).to receive(:system)
        .with(dir/"bin/pip", "install", "-v", "--no-deps",
              "--no-binary", ":all:", "--ignore-installed", "bar")
        .and_return(true)

      virtualenv.pip_install ["foo", "bar"]
    end

    it "accepts a Resource" do
      res = Resource.new("test")

      expect(res).to receive(:stage).and_yield
      expect(formula).to receive(:system)
        .with(dir/"bin/pip", "install", "-v", "--no-deps",
              "--no-binary", ":all:", "--ignore-installed", Pathname.pwd)
        .and_return(true)

      virtualenv.pip_install res
    end
  end

  describe "#pip_install_and_link" do
    let(:src_bin) { dir/"bin" }
    let(:dest_bin) { formula.bin }

    it "can link scripts" do
      src_bin.mkpath

      expect(src_bin/"kilroy").not_to exist
      expect(dest_bin/"kilroy").not_to exist

      FileUtils.touch src_bin/"irrelevant"
      bin_before = Dir.glob(src_bin/"*")
      FileUtils.touch src_bin/"kilroy"
      bin_after = Dir.glob(src_bin/"*")

      expect(virtualenv).to receive(:pip_install).with("foo")
      expect(Dir).to receive(:[]).with(src_bin/"*").twice.and_return(bin_before, bin_after)

      virtualenv.pip_install_and_link "foo"

      expect(src_bin/"kilroy").to exist
      expect(dest_bin/"kilroy").to exist
      expect(dest_bin/"kilroy").to be_a_symlink
      expect((src_bin/"kilroy").realpath).to eq((dest_bin/"kilroy").realpath)
      expect(dest_bin/"irrelevant").not_to exist
    end
  end
end
