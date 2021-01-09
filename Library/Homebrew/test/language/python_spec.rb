# typed: false
# frozen_string_literal: true

require "language/python"
require "resource"
require "utils/shebang"

describe Language::Python, :needs_python do
  describe "#major_minor_version" do
    it "returns a Version for Python 2" do
      expect(subject).to receive(:major_minor_version).and_return(Version)
      subject.major_minor_version("python")
    end
  end

  describe "#site_packages" do
    it "gives a different location between PyPy and Python 2" do
      expect(subject.site_packages("python")).not_to eql(subject.site_packages("pypy"))
    end
  end

  describe "#homebrew_site_packages" do
    it "returns the Homebrew site packages location" do
      expect(subject).to receive(:site_packages).and_return(Pathname)
      subject.site_packages("python")
    end
  end

  describe "#user_site_packages" do
    it "can determine user site packages location" do
      expect(subject).to receive(:user_site_packages).and_return(Pathname)
      subject.user_site_packages("python")
    end
  end
end

describe Language::Python::Shebang do
  let(:file) { Tempfile.new("python-shebang") }
  let(:python_f) do
    formula "python" do
      url "https://brew.sh/python-1.0.tgz"
    end
  end
  let(:f) do
    formula "foo" do
      url "https://brew.sh/foo-1.0.tgz"

      depends_on "python"
    end
  end

  before do
    file.write <<~EOS
      #!/usr/bin/env python3
      a
      b
      c
    EOS
    file.flush
  end

  after { file.unlink }

  describe "#detected_python_shebang" do
    it "can be used to replace Python shebangs" do
      expect(Formulary).to receive(:factory).with(python_f.name).and_return(python_f)
      Utils::Shebang.rewrite_shebang described_class.detected_python_shebang(f), file

      expect(File.read(file)).to eq <<~EOS
        #!#{HOMEBREW_PREFIX}/opt/python/bin/python3
        a
        b
        c
      EOS
    end
  end
end

describe Language::Python::Virtualenv::Virtualenv do
  subject { described_class.new(formula, dir, "python") }

  let(:dir) { mktmpdir }

  let(:resource) { double("resource", stage: true) }
  let(:formula_bin) { dir/"formula_bin" }
  let(:formula) { double("formula", resource: resource, bin: formula_bin) }

  describe "#create" do
    it "creates a venv" do
      expect(formula).to receive(:system).with("python", "-m", "venv", dir)
      subject.create
    end
  end

  describe "#pip_install" do
    it "accepts a string" do
      expect(formula).to receive(:system)
        .with(dir/"bin/pip", "install", "-v", "--no-deps",
              "--no-binary", ":all:", "--no-user", "--ignore-installed", "foo")
        .and_return(true)
      subject.pip_install "foo"
    end

    it "accepts a multi-line strings" do
      expect(formula).to receive(:system)
        .with(dir/"bin/pip", "install", "-v", "--no-deps",
              "--no-binary", ":all:", "--no-user", "--ignore-installed", "foo", "bar")
        .and_return(true)

      subject.pip_install <<~EOS
        foo
        bar
      EOS
    end

    it "accepts an array" do
      expect(formula).to receive(:system)
        .with(dir/"bin/pip", "install", "-v", "--no-deps",
              "--no-binary", ":all:", "--no-user", "--ignore-installed", "foo")
        .and_return(true)

      expect(formula).to receive(:system)
        .with(dir/"bin/pip", "install", "-v", "--no-deps",
              "--no-binary", ":all:", "--no-user", "--ignore-installed", "bar")
        .and_return(true)

      subject.pip_install ["foo", "bar"]
    end

    it "accepts a Resource" do
      res = Resource.new("test")

      expect(res).to receive(:stage).and_yield
      expect(formula).to receive(:system)
        .with(dir/"bin/pip", "install", "-v", "--no-deps",
              "--no-binary", ":all:", "--no-user", "--ignore-installed", Pathname.pwd)
        .and_return(true)

      subject.pip_install res
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

      expect(subject).to receive(:pip_install).with("foo")
      expect(Dir).to receive(:[]).with(src_bin/"*").twice.and_return(bin_before, bin_after)

      subject.pip_install_and_link "foo"

      expect(src_bin/"kilroy").to exist
      expect(dest_bin/"kilroy").to exist
      expect(dest_bin/"kilroy").to be_a_symlink
      expect((src_bin/"kilroy").realpath).to eq((dest_bin/"kilroy").realpath)
      expect(dest_bin/"irrelevant").not_to exist
    end
  end
end
