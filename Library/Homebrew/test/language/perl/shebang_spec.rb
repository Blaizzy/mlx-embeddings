# typed: false
# frozen_string_literal: true

require "language/perl"
require "utils/shebang"

describe Language::Perl::Shebang do
  let(:file) { Tempfile.new("perl-shebang") }
  let(:perl_f) do
    formula "perl" do
      url "https://brew.sh/perl-1.0.tgz"
    end
  end
  let(:f) do
    formula "foo" do
      url "https://brew.sh/foo-1.0.tgz"

      uses_from_macos "perl"
    end
  end

  before do
    file.write <<~EOS
      #!/usr/bin/env perl
      a
      b
      c
    EOS
    file.flush
  end

  after { file.unlink }

  describe "#detected_perl_shebang" do
    it "can be used to replace Perl shebangs" do
      allow(Formulary).to receive(:factory).with(perl_f.name).and_return(perl_f)
      Utils::Shebang.rewrite_shebang described_class.detected_perl_shebang(f), file

      expected_shebang = if OS.mac?
        "/usr/bin/perl#{MacOS.preferred_perl_version}"
      else
        HOMEBREW_PREFIX/"opt/perl/bin/perl"
      end

      expect(File.read(file)).to eq <<~EOS
        #!#{expected_shebang}
        a
        b
        c
      EOS
    end
  end
end
