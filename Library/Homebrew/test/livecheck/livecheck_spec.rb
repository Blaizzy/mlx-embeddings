# typed: false
# frozen_string_literal: true

require "livecheck/livecheck"

describe Homebrew::Livecheck do
  subject(:livecheck) { described_class }

  let(:cask_url) { "https://brew.sh/test-0.0.1.dmg" }
  let(:head_url) { "https://github.com/Homebrew/brew.git" }
  let(:homepage_url) { "https://brew.sh" }
  let(:livecheck_url) { "https://formulae.brew.sh/api/formula/ruby.json" }
  let(:stable_url) { "https://brew.sh/test-0.0.1.tgz" }

  let(:f) do
    formula("test") do
      desc "Test formula"
      homepage "https://brew.sh"
      url "https://brew.sh/test-0.0.1.tgz"
      head "https://github.com/Homebrew/brew.git"

      livecheck do
        url "https://formulae.brew.sh/api/formula/ruby.json"
        regex(/"stable":"(\d+(?:\.\d+)+)"/i)
      end
    end
  end

  let(:c) do
    Cask::CaskLoader.load(+<<-RUBY)
      cask "test" do
        version "0.0.1,2"

        url "https://brew.sh/test-0.0.1.dmg"
        name "Test"
        desc "Test cask"
        homepage "https://brew.sh"

        livecheck do
          url "https://formulae.brew.sh/api/formula/ruby.json"
          regex(/"stable":"(\d+(?:\.\d+)+)"/i)
        end
      end
    RUBY
  end

  describe "::formula_name" do
    it "returns the name of the formula" do
      expect(livecheck.formula_name(f)).to eq("test")
    end

    it "returns the full name" do
      expect(livecheck.formula_name(f, full_name: true)).to eq("test")
    end
  end

  describe "::cask_name" do
    it "returns the token of the cask" do
      expect(livecheck.cask_name(c)).to eq("test")
    end

    it "returns the full name of the cask" do
      expect(livecheck.cask_name(c, full_name: true)).to eq("test")
    end
  end

  describe "::status_hash" do
    it "returns a hash containing the livecheck status" do
      expect(livecheck.status_hash(f, "error", ["Unable to get versions"]))
        .to eq({
          formula:  "test",
          status:   "error",
          messages: ["Unable to get versions"],
          meta:     {
            livecheckable: true,
          },
        })
    end
  end

  describe "::livecheck_url_to_string" do
    let(:f_livecheck_url) do
      formula("test_livecheck_url") do
        desc "Test Livecheck URL formula"
        homepage "https://brew.sh"
        url "https://brew.sh/test-0.0.1.tgz"
        head "https://github.com/Homebrew/brew.git"
      end
    end

    let(:c_livecheck_url) do
      Cask::CaskLoader.load(+<<-RUBY)
        cask "test_livecheck_url" do
          version "0.0.1,2"

          url "https://brew.sh/test-0.0.1.dmg"
          name "Test"
          desc "Test Livecheck URL cask"
          homepage "https://brew.sh"
        end
      RUBY
    end

    it "returns a URL string when given a livecheck_url string" do
      f_livecheck_url.livecheck.url(livecheck_url)
      expect(livecheck.livecheck_url_to_string(livecheck_url, f_livecheck_url)).to eq(livecheck_url)
    end

    it "returns a URL symbol when given a valid livecheck_url symbol" do
      f_livecheck_url.livecheck.url(:head)
      expect(livecheck.livecheck_url_to_string(head_url, f_livecheck_url)).to eq(head_url)

      f_livecheck_url.livecheck.url(:homepage)
      expect(livecheck.livecheck_url_to_string(homepage_url, f_livecheck_url)).to eq(homepage_url)

      c_livecheck_url.livecheck.url(:homepage)
      expect(livecheck.livecheck_url_to_string(homepage_url, c_livecheck_url)).to eq(homepage_url)

      f_livecheck_url.livecheck.url(:stable)
      expect(livecheck.livecheck_url_to_string(stable_url, f_livecheck_url)).to eq(stable_url)

      c_livecheck_url.livecheck.url(:url)
      expect(livecheck.livecheck_url_to_string(cask_url, c_livecheck_url)).to eq(cask_url)
    end

    it "returns nil when not given a string or valid symbol" do
      expect(livecheck.livecheck_url_to_string(nil, f_livecheck_url)).to eq(nil)
      expect(livecheck.livecheck_url_to_string(nil, c_livecheck_url)).to eq(nil)
      expect(livecheck.livecheck_url_to_string(:invalid_symbol, f_livecheck_url)).to eq(nil)
      expect(livecheck.livecheck_url_to_string(:invalid_symbol, c_livecheck_url)).to eq(nil)
    end
  end

  describe "::checkable_urls" do
    it "returns the list of URLs to check" do
      expect(livecheck.checkable_urls(f)).to eq([stable_url, head_url, homepage_url])
      expect(livecheck.checkable_urls(c)).to eq([cask_url, homepage_url])
    end
  end

  describe "::preprocess_url" do
    let(:github_git_url_with_extension) { "https://github.com/Homebrew/brew.git" }

    it "returns the unmodified URL for an unparseable URL" do
      # Modeled after the `head` URL in the `ncp` formula
      expect(livecheck.preprocess_url(":something:cvs:@cvs.brew.sh:/cvs"))
        .to eq(":something:cvs:@cvs.brew.sh:/cvs")
    end

    it "returns the unmodified URL for a GitHub URL ending in .git" do
      expect(livecheck.preprocess_url(github_git_url_with_extension))
        .to eq(github_git_url_with_extension)
    end

    it "returns the Git repository URL for a GitHub URL not ending in .git" do
      expect(livecheck.preprocess_url("https://github.com/Homebrew/brew"))
        .to eq(github_git_url_with_extension)
    end

    it "returns the unmodified URL for a GitHub /releases/latest URL" do
      expect(livecheck.preprocess_url("https://github.com/Homebrew/brew/releases/latest"))
        .to eq("https://github.com/Homebrew/brew/releases/latest")
    end

    it "returns the Git repository URL for a GitHub AWS URL" do
      expect(livecheck.preprocess_url("https://github.s3.amazonaws.com/downloads/Homebrew/brew/1.0.0.tar.gz"))
        .to eq(github_git_url_with_extension)
    end

    it "returns the Git repository URL for a github.com/downloads/... URL" do
      expect(livecheck.preprocess_url("https://github.com/downloads/Homebrew/brew/1.0.0.tar.gz"))
        .to eq(github_git_url_with_extension)
    end

    it "returns the Git repository URL for a GitHub tag archive URL" do
      expect(livecheck.preprocess_url("https://github.com/Homebrew/brew/archive/1.0.0.tar.gz"))
        .to eq(github_git_url_with_extension)
    end

    it "returns the Git repository URL for a GitHub release archive URL" do
      expect(livecheck.preprocess_url("https://github.com/Homebrew/brew/releases/download/1.0.0/brew-1.0.0.tar.gz"))
        .to eq(github_git_url_with_extension)
    end

    it "returns the Git repository URL for a gitlab.com archive URL" do
      expect(livecheck.preprocess_url("https://gitlab.com/Homebrew/brew/-/archive/1.0.0/brew-1.0.0.tar.gz"))
        .to eq("https://gitlab.com/Homebrew/brew.git")
    end

    it "returns the Git repository URL for a self-hosted GitLab archive URL" do
      expect(livecheck.preprocess_url("https://brew.sh/Homebrew/brew/-/archive/1.0.0/brew-1.0.0.tar.gz"))
        .to eq("https://brew.sh/Homebrew/brew.git")
    end

    it "returns the Git repository URL for a Codeberg archive URL" do
      expect(livecheck.preprocess_url("https://codeberg.org/Homebrew/brew/archive/brew-1.0.0.tar.gz"))
        .to eq("https://codeberg.org/Homebrew/brew.git")
    end

    it "returns the Git repository URL for a Gitea archive URL" do
      expect(livecheck.preprocess_url("https://gitea.com/Homebrew/brew/archive/brew-1.0.0.tar.gz"))
        .to eq("https://gitea.com/Homebrew/brew.git")
    end

    it "returns the Git repository URL for an Opendev archive URL" do
      expect(livecheck.preprocess_url("https://opendev.org/Homebrew/brew/archive/brew-1.0.0.tar.gz"))
        .to eq("https://opendev.org/Homebrew/brew.git")
    end

    it "returns the Git repository URL for a tildegit archive URL" do
      expect(livecheck.preprocess_url("https://tildegit.org/Homebrew/brew/archive/brew-1.0.0.tar.gz"))
        .to eq("https://tildegit.org/Homebrew/brew.git")
    end

    it "returns the Git repository URL for a LOL Git archive URL" do
      expect(livecheck.preprocess_url("https://lolg.it/Homebrew/brew/archive/brew-1.0.0.tar.gz"))
        .to eq("https://lolg.it/Homebrew/brew.git")
    end

    it "returns the Git repository URL for a sourcehut archive URL" do
      expect(livecheck.preprocess_url("https://git.sr.ht/~Homebrew/brew/archive/1.0.0.tar.gz"))
        .to eq("https://git.sr.ht/~Homebrew/brew")
    end
  end
end
