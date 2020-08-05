# frozen_string_literal: true

require "rubocops/rubocop-cask"
require "test/rubocops/cask/shared_examples/cask_cop"

describe RuboCop::Cop::Cask::Desc do
  include CaskCop

  subject(:cop) { described_class.new }

  context "with incorrect `desc` stanza" do
    let(:source) {
      <<~RUBY
        cask "foo" do
          desc "A bar program"
        end
      RUBY
    }
    let(:correct_source) {
      <<~RUBY
        cask "foo" do
          desc "Bar program"
        end
      RUBY
    }
    let(:expected_offenses) do
      [{
        message:  "Description shouldn't start with an indefinite article, i.e. \"A\".",
        severity: :convention,
        line:     2,
        column:   8,
        source:   "A",
      }]
    end

    include_examples "reports offenses"

    include_examples "autocorrects source"
  end

  context "with correct `desc` stanza" do
    let(:source) {
      <<~RUBY
        cask "foo" do
          desc "Bar program"
        end
      RUBY
    }

    include_examples "does not report any offenses"
  end
end
