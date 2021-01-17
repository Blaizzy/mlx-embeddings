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
    it "returns the revision at HEAD if repo parameter is specified" do
      expect(described_class.git_head(HOMEBREW_CACHE)).to eq(head_revision)
      expect(described_class.git_head(HOMEBREW_CACHE, length: 5)).to eq(head_revision[0...5])
    end

    it "returns the revision at HEAD if repo parameter is omitted" do
      HOMEBREW_CACHE.cd do
        expect(described_class.git_head).to eq(head_revision)
        expect(described_class.git_head(length: 5)).to eq(head_revision[0...5])
      end
    end

    context "when directory is not a Git repository" do
      it "returns nil if `safe` parameter is `false`" do
        expect(described_class.git_head(TEST_TMPDIR, safe: false)).to eq(nil)
      end

      it "raises an error if `safe` parameter is `true`" do
        expect { described_class.git_head(TEST_TMPDIR, safe: true) }
          .to raise_error("Not a Git repository: #{TEST_TMPDIR}")
      end
    end

    context "when Git is unavailable" do
      before do
        allow(Utils::Git).to receive(:available?).and_return(false)
      end

      it "returns nil if `safe` parameter is `false`" do
        expect(described_class.git_head(HOMEBREW_CACHE, safe: false)).to eq(nil)
      end

      it "raises an error if `safe` parameter is `true`" do
        expect { described_class.git_head(HOMEBREW_CACHE, safe: true) }
          .to raise_error("Git is unavailable")
      end
    end
  end

  describe ".git_short_head" do
    it "returns the short revision at HEAD if repo parameter is specified" do
      expect(described_class.git_short_head(HOMEBREW_CACHE)).to eq(short_head_revision)
      expect(described_class.git_short_head(HOMEBREW_CACHE, length: 5)).to eq(head_revision[0...5])
    end

    it "returns the short revision at HEAD if repo parameter is omitted" do
      HOMEBREW_CACHE.cd do
        expect(described_class.git_short_head).to eq(short_head_revision)
        expect(described_class.git_short_head(length: 5)).to eq(head_revision[0...5])
      end
    end

    context "when directory is not a Git repository" do
      it "returns nil if `safe` parameter is `false`" do
        expect(described_class.git_short_head(TEST_TMPDIR, safe: false)).to eq(nil)
      end

      it "raises an error if `safe` parameter is `true`" do
        expect { described_class.git_short_head(TEST_TMPDIR, safe: true) }
          .to raise_error("Not a Git repository: #{TEST_TMPDIR}")
      end
    end

    context "when Git is unavailable" do
      before do
        allow(Utils::Git).to receive(:available?).and_return(false)
      end

      it "returns nil if `safe` parameter is `false`" do
        expect(described_class.git_short_head(HOMEBREW_CACHE, safe: false)).to eq(nil)
      end

      it "raises an error if `safe` parameter is `true`" do
        expect { described_class.git_short_head(HOMEBREW_CACHE, safe: true) }
          .to raise_error("Git is unavailable")
      end
    end
  end
end
