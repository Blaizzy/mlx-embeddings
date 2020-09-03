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

  let(:f_deprecated) do
    formula("test_deprecated") do
      desc "Deprecated test formula"
      homepage "https://brew.sh"
      url "https://brew.sh/test-0.0.1.tgz"
      deprecate! because: :unmaintained
    end
  end

  let(:f_gist) do
    formula("test_gist") do
      desc "Gist test formula"
      homepage "https://brew.sh"
      url "https://gist.github.com/Homebrew/0000000000"
    end
  end

  let(:f_head_only) do
    formula("test_head_only") do
      desc "HEAD-only test formula"
      homepage "https://brew.sh"
      head "https://github.com/Homebrew/brew.git"
    end
  end

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

  let(:f_versioned) do
    formula("test@0.0.1") do
      desc "Versioned test formula"
      homepage "https://brew.sh"
      url "https://brew.sh/test-0.0.1.tgz"
    end
  end

  let(:args) { double("livecheck_args", full_name?: false, json?: false, quiet?: false, verbose?: true) }

  describe "::formula_name" do
    it "returns the name of the formula" do
      expect(livecheck.formula_name(f, args: args)).to eq("test")
    end

    it "returns the full name" do
      allow(args).to receive(:full_name?).and_return(true)

      expect(livecheck.formula_name(f, args: args)).to eq("test")
    end
  end

  describe "::status_hash" do
    it "returns a hash containing the livecheck status" do
      expect(livecheck.status_hash(f, "error", ["Unable to get versions"], args: args))
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
    it "skips a deprecated formula without a livecheckable" do
      expect { livecheck.skip_conditions(f_deprecated, args: args) }
        .to output("test_deprecated : deprecated\n").to_stdout
        .and not_to_output.to_stderr
    end

    it "skips a versioned formula without a livecheckable" do
      expect { livecheck.skip_conditions(f_versioned, args: args) }
        .to output("test@0.0.1 : versioned\n").to_stdout
        .and not_to_output.to_stderr
    end

    it "skips a HEAD-only formula if not installed" do
      expect { livecheck.skip_conditions(f_head_only, args: args) }
        .to output("test_head_only : HEAD only formula must be installed to be livecheckable\n").to_stdout
        .and not_to_output.to_stderr
    end

    it "skips a formula with a GitHub Gist stable URL" do
      expect { livecheck.skip_conditions(f_gist, args: args) }
        .to output("test_gist : skipped - Stable URL is a GitHub Gist\n").to_stdout
        .and not_to_output.to_stderr
    end

    it "skips a formula with a skip livecheckable" do
      expect { livecheck.skip_conditions(f_skip, args: args) }
        .to output("test_skip : skipped - Not maintained\n").to_stdout
        .and not_to_output.to_stderr
    end

    it "returns false for a non-skippable formula" do
      expect(livecheck.skip_conditions(f, args: args)).to eq(false)
    end
  end

  describe "::checkable_urls" do
    it "returns the list of URLs to check" do
      expect(livecheck.checkable_urls(f))
        .to eq(
          ["https://github.com/Homebrew/brew.git", "https://brew.sh/test-0.0.1.tgz", "https://brew.sh"],
        )
    end
  end

  describe "::preprocess_url" do
    let(:url) { "https://github.s3.amazonaws.com/downloads/Homebrew/brew/1.0.0.tar.gz" }

    it "returns the preprocessed URL for livecheck to use" do
      expect(livecheck.preprocess_url(url))
        .to eq("https://github.com/Homebrew/brew.git")
    end
  end
end
