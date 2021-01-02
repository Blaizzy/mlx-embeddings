# typed: false
# Frozen_string_literal: true

require "livecheck/livecheck"

describe Homebrew::Livecheck do
  subject(:livecheck) { described_class }

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

        url "https://brew.sh/test-0.0.1.tgz"
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

  describe "::skip_conditions" do
    context "a deprecated formula without a livecheckable" do
      let(:f_deprecated) do
        formula("test_deprecated") do
          desc "Deprecated test formula"
          homepage "https://brew.sh"
          url "https://brew.sh/test-0.0.1.tgz"
          deprecate! date: "2020-06-25", because: :unmaintained
        end
      end

      it "skips" do
        expect { livecheck.skip_conditions(f_deprecated) }
          .to output("test_deprecated : deprecated\n").to_stdout
          .and not_to_output.to_stderr
      end
    end

    context "a discontinued cask without a livecheckable" do
      let(:c_discontinued) do
        Cask::CaskLoader.load(+<<-RUBY)
          cask "test_discontinued" do
            version "0.0.1"
            sha256 :no_check

            url "https://brew.sh/test-0.0.1.tgz"
            name "Test Discontinued"
            desc "Discontinued test cask"
            homepage "https://brew.sh"

            caveats do
              discontinued
            end
          end
        RUBY
      end

      it "skips" do
        expect { livecheck.skip_conditions(c_discontinued) }
          .to output("test_discontinued : discontinued\n").to_stdout
          .and not_to_output.to_stderr
      end
    end

    context "a disabled formula without a livecheckable" do
      let(:f_disabled) do
        formula("test_disabled") do
          desc "Disabled test formula"
          homepage "https://brew.sh"
          url "https://brew.sh/test-0.0.1.tgz"
          disable! date: "2020-06-25", because: :unmaintained
        end
      end

      it "skips" do
        expect { livecheck.skip_conditions(f_disabled) }
          .to output("test_disabled : disabled\n").to_stdout
          .and not_to_output.to_stderr
      end
    end

    context "a versioned formula without a livecheckable" do
      let(:f_versioned) do
        formula("test@0.0.1") do
          desc "Versioned test formula"
          homepage "https://brew.sh"
          url "https://brew.sh/test-0.0.1.tgz"
        end
      end

      it "skips" do
        expect { livecheck.skip_conditions(f_versioned) }
          .to output("test@0.0.1 : versioned\n").to_stdout
          .and not_to_output.to_stderr
      end
    end

    context "a cask containing `version :latest` without a livecheckable" do
      let(:c_latest) do
        Cask::CaskLoader.load(+<<-RUBY)
          cask "test_latest" do
            version :latest
            sha256 :no_check

            url "https://brew.sh/test-0.0.1.tgz"
            name "Test Latest"
            desc "Latest test cask"
            homepage "https://brew.sh"
          end
        RUBY
      end

      it "skips" do
        expect { livecheck.skip_conditions(c_latest) }
          .to output("test_latest : latest\n").to_stdout
          .and not_to_output.to_stderr
      end
    end

    context "a cask containing an unversioned URL without a livecheckable" do
      # `URL#unversioned?` doesn't work properly when using the
      # `Cask::CaskLoader.load` setup above, so we use `Cask::Cask.new` instead.
      let(:c_unversioned) do
        Cask::Cask.new "test_unversioned" do
          version "1.2.3"
          sha256 :no_check

          url "https://brew.sh/test.tgz"
          name "Test Unversioned"
          desc "Unversioned test cask"
          homepage "https://brew.sh"
        end
      end

      it "skips" do
        expect { livecheck.skip_conditions(c_unversioned) }
          .to output("test_unversioned : unversioned\n").to_stdout
          .and not_to_output.to_stderr
      end
    end

    context "a HEAD-only formula that is not installed" do
      let(:f_head_only) do
        formula("test_head_only") do
          desc "HEAD-only test formula"
          homepage "https://brew.sh"
          head "https://github.com/Homebrew/brew.git"
        end
      end

      it "skips " do
        expect { livecheck.skip_conditions(f_head_only) }
          .to output("test_head_only : HEAD only formula must be installed to be livecheckable\n").to_stdout
          .and not_to_output.to_stderr
      end
    end

    context "a formula with a GitHub Gist stable URL" do
      let(:f_gist) do
        formula("test_gist") do
          desc "Gist test formula"
          homepage "https://brew.sh"
          url "https://gist.github.com/Homebrew/0000000000"
        end
      end

      it "skips" do
        expect { livecheck.skip_conditions(f_gist) }
          .to output("test_gist : skipped - Stable URL is a GitHub Gist\n").to_stdout
          .and not_to_output.to_stderr
      end
    end

    context "a formula with a Google Code Archive stable URL" do
      let(:f_google_code_archive) do
        formula("test_google_code_archive") do
          desc "Google Code Archive test formula"
          homepage "https://brew.sh"
          url "https://storage.googleapis.com/google-code-archive-downloads/v2/code.google.com/brew/brew-1.0.0.tar.gz"
        end
      end

      it "skips" do
        expect { livecheck.skip_conditions(f_google_code_archive) }
          .to output("test_google_code_archive : skipped - Stable URL is from Google Code Archive\n").to_stdout
          .and not_to_output.to_stderr
      end
    end

    context "a formula with a `livecheck` block containing `skip`" do
      let(:f_skip) do
        formula("test_skip") do
          desc "Skipped test formula"
          homepage "https://brew.sh"
          url "https://brew.sh/test-0.0.1.tgz"

          livecheck do
            skip "Not maintained"
          end
        end
      end

      it "skips" do
        expect { livecheck.skip_conditions(f_skip) }
          .to output("test_skip : skipped - Not maintained\n").to_stdout
          .and not_to_output.to_stderr
      end
    end

    it "returns false for a non-skippable formula" do
      expect(livecheck.skip_conditions(f)).to eq(false)
    end

    it "returns false for a non-skippable cask" do
      expect(livecheck.skip_conditions(c)).to eq(false)
    end
  end

  describe "::checkable_urls" do
    it "returns the list of URLs to check" do
      expect(livecheck.checkable_urls(f))
        .to eq(
          ["https://github.com/Homebrew/brew.git", "https://brew.sh/test-0.0.1.tgz", "https://brew.sh"],
        )
      expect(livecheck.checkable_urls(c)).to eq(["https://brew.sh/test-0.0.1.tgz", "https://brew.sh"])
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
