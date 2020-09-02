# frozen_string_literal: true

require "style"

describe Homebrew::Style do
  around do |example|
    FileUtils.ln_s HOMEBREW_LIBRARY_PATH, HOMEBREW_LIBRARY/"Homebrew"
    FileUtils.ln_s HOMEBREW_LIBRARY_PATH.parent/".rubocop.yml", HOMEBREW_LIBRARY/".rubocop.yml"

    example.run
  ensure
    FileUtils.rm_f HOMEBREW_LIBRARY/"Homebrew"
    FileUtils.rm_f HOMEBREW_LIBRARY/".rubocop.yml"
  end

  before do
    allow(Homebrew).to receive(:install_bundler_gems!)
  end

  describe ".check_style_json" do
    let(:dir) { mktmpdir }

    it "returns offenses when RuboCop reports offenses" do
      formula = dir/"my-formula.rb"

      formula.write <<~'EOS'
        class MyFormula < Formula

        end
      EOS

      style_offenses = described_class.check_style_json([formula])

      expect(style_offenses.for_path(formula.realpath).map(&:message))
        .to include("Extra empty line detected at class body beginning.")
    end

    it "corrected offense output format" do
      formula = dir/"my-formula-2.rb"

      formula.write <<~EOS
        class MyFormula2 < Formula
          desc "Test formula"
          homepage "https://foo.org"
          url "https://foo.org/foo-1.7.5.tgz"
          sha256 "cc692fb9dee0cc288757e708fc1a3b6b56ca1210ca181053a371cb11746969da"

          depends_on "foo"
          depends_on "bar-config" => :build

          test do
            assert_equal 5, 5
          end
        end
      EOS
      style_offenses = described_class.check_style_json(
        [formula],
        fix: true, only_cops: ["FormulaAudit/DependencyOrder"],
      )
      offense_string = style_offenses.for_path(formula.realpath).first.to_s
      expect(offense_string).to match(/\[Corrected\]/)
    end
  end

  describe ".check_style_and_print" do
    let(:dir) { mktmpdir }

    it "returns false for conforming file with only audit-level violations" do
      # This file is known to use non-rocket hashes and other things that trigger audit,
      # but not regular, cop violations
      target_file = HOMEBREW_LIBRARY_PATH/"utils.rb"

      style_result = described_class.check_style_and_print([target_file])

      expect(style_result).to eq true
    end
  end
end
