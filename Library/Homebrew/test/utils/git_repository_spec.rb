# typed: false
# frozen_string_literal: true

require "utils/git_repository"

describe Utils do
  before do
    HOMEBREW_CACHE.cd do
      system "git", "init"
      Pathname("README.md").write("README")
      system "git", "add", "README.md"
      system "git", "commit", "-m", "File added"
    end
  end

  let(:head_revision) { HOMEBREW_CACHE.cd { `git rev-parse HEAD`.chomp } }
  let(:short_head_revision) { HOMEBREW_CACHE.cd { `git rev-parse --short HEAD`.chomp } }

  describe ".git_head" do
    it "returns the revision at HEAD" do
      expect(described_class.git_head(HOMEBREW_CACHE)).to eq(head_revision)
      expect(described_class.git_head(HOMEBREW_CACHE, length: 5)).to eq(head_revision[0...5])
      HOMEBREW_CACHE.cd do
        expect(described_class.git_head).to eq(head_revision)
        expect(described_class.git_head(length: 5)).to eq(head_revision[0...5])
      end
    end
  end

  describe ".git_short_head" do
    it "returns the short revision at HEAD" do
      expect(described_class.git_short_head(HOMEBREW_CACHE)).to eq(short_head_revision)
      expect(described_class.git_short_head(HOMEBREW_CACHE, length: 5)).to eq(head_revision[0...5])
      HOMEBREW_CACHE.cd do
        expect(described_class.git_short_head).to eq(short_head_revision)
        expect(described_class.git_short_head(length: 5)).to eq(head_revision[0...5])
      end
    end
  end
end
