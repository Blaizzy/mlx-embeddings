# typed: false
# frozen_string_literal: true

require "rubocops/rubocop-cask"
require "test/rubocops/cask/shared_examples/cask_cop"

describe RuboCop::Cop::Cask::UrlLegacyCommaSeparators do
  include CaskCop

  subject(:cop) { described_class.new }

  context "when url version interpolation does not include version.before_comma or version.after_comma" do
    let(:source) do
      <<-CASK.undent
        cask 'foo' do
          version '1.1'
          url 'https://foo.brew.sh/foo-\#{version}.dmg'
        end
      CASK
    end

    include_examples "does not report any offenses"
  end

  context "when the url uses csv" do
    let(:source) do
      <<-CASK.undent
        cask 'foo' do
          version '1.1,111'
          url 'https://foo.brew.sh/foo-\#{version.csv.first}.dmg'
        end
      CASK
    end

    include_examples "does not report any offenses"
  end

  context "when the url uses version.before_comma" do
    let(:source) do
      <<-CASK.undent
        cask 'foo' do
          version '1.1,111'
          url 'https://foo.brew.sh/foo-\#{version.before_comma}.dmg'
        end
      CASK
    end
    let(:correct_source) do
      <<-CASK.undent
        cask 'foo' do
          version '1.1,111'
          url 'https://foo.brew.sh/foo-\#{version.csv.first}.dmg'
        end
      CASK
    end
    let(:expected_offenses) do
      [{
        message:  "Use 'version.csv.first' instead of 'version.before_comma' "\
                  "and 'version.csv.second' instead of 'version.after_comma'",
        severity: :convention,
        line:     3,
        column:   6,
        source:   "'https://foo.brew.sh/foo-\#{version.before_comma}.dmg'",
      }]
    end

    include_examples "reports offenses"

    include_examples "autocorrects source"
  end

  context "when the url uses version.after_comma" do
    let(:source) do
      <<-CASK.undent
        cask 'foo' do
          version '1.1,111'
          url 'https://foo.brew.sh/foo-\#{version.after_comma}.dmg'
        end
      CASK
    end
    let(:correct_source) do
      <<-CASK.undent
        cask 'foo' do
          version '1.1,111'
          url 'https://foo.brew.sh/foo-\#{version.csv.second}.dmg'
        end
      CASK
    end
    let(:expected_offenses) do
      [{
        message:  "Use 'version.csv.first' instead of 'version.before_comma' "\
                  "and 'version.csv.second' instead of 'version.after_comma'",
        severity: :convention,
        line:     3,
        column:   6,
        source:   "'https://foo.brew.sh/foo-\#{version.after_comma}.dmg'",
      }]
    end

    include_examples "reports offenses"

    include_examples "autocorrects source"
  end
end
