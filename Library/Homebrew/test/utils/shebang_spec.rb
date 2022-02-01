# typed: false
# frozen_string_literal: true

require "utils/shebang"

describe Utils::Shebang do
  let(:file) { Tempfile.new("shebang") }

  before do
    file.write "#!/usr/bin/python"
    file.flush
  end

  after { file.unlink }

  describe "rewrite_shebang" do
    it "rewrites a shebang" do
      rewrite_info = Utils::Shebang::RewriteInfo.new(/^#!.*python$/, 25, "new_shebang")
      described_class.rewrite_shebang rewrite_info, file
      expect(File.read(file)).to eq "#!new_shebang"
    end

    it "raises an error if no rewriting occurs" do
      rewrite_info = Utils::Shebang::RewriteInfo.new(/^#!.*perl$/, 25, "new_shebang")
      expect {
        described_class.rewrite_shebang rewrite_info, file
      }.to raise_error("No matching files found to rewrite shebang")
    end
  end
end
