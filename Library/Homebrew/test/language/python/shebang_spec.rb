# typed: false
# frozen_string_literal: true

require "language/python"
require "utils/shebang"

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
