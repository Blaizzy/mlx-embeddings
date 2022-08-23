# typed: false
# frozen_string_literal: true

require "language/python"
require "utils/shebang"

describe Language::Python::Shebang do
  let(:file) { Tempfile.new("python-shebang") }
  let(:python_f) do
    formula "python@3.11" do
      url "https://brew.sh/python-1.0.tgz"
    end
  end
  let(:f) do
    formula "foo" do
      url "https://brew.sh/foo-1.0.tgz"

      depends_on "python@3.11"
    end
  end

  before do
    file.write <<~EOS
      #!/usr/bin/python2
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
      Utils::Shebang.rewrite_shebang described_class.detected_python_shebang(f, use_python_from_path: false), file

      expect(File.read(file)).to eq <<~EOS
        #!#{HOMEBREW_PREFIX}/opt/python@3.11/bin/python3.11
        a
        b
        c
      EOS
    end

    it "can be pointed to a `python3` in PATH" do
      Utils::Shebang.rewrite_shebang described_class.detected_python_shebang(f, use_python_from_path: true), file

      expect(File.read(file)).to eq <<~EOS
        #!/usr/bin/env python3
        a
        b
        c
      EOS
    end
  end
end
