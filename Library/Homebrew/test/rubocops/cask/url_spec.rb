# typed: false
# frozen_string_literal: true

require "rubocops/rubocop-cask"
require "test/rubocops/cask/shared_examples/cask_cop"

describe RuboCop::Cop::Cask::Url do
  include CaskCop

  subject(:cop) { described_class.new }

  context "when url 'verified' value does not start with a protocol" do
    let(:source) do
      <<~CASK
        cask "foo" do
          url "https://example.com/download/foo-v1.2.0.dmg",
            verified: "example.com"
        end
      CASK
    end

    include_examples "does not report any offenses"
  end

  context "when url 'verified' value starts with a protocol" do
    let(:source) do
      <<~CASK
        cask "foo" do
          url "https://example.com/download/foo-v1.2.0.dmg",
            verified: "https://example.com"
        end
      CASK
    end

    let(:expected_offenses) do
      [{
        message:  "Verified URL parameter value should not start with https:// or http://.",
        severity: :convention,
        line:     3,
        column:   14,
        source:   "\"https://example.com\"",
      }]
    end

    let(:correct_source) do
      <<~CASK
        cask "foo" do
          url "https://example.com/download/foo-v1.2.0.dmg",
            verified: "example.com"
        end
      CASK
    end

    include_examples "reports offenses"
    include_examples "autocorrects source"
  end

  context "when url 'verified' value has a path component that ends with a /" do
    let(:source) do
      <<~CASK
        cask "foo" do
          url "https://example.com/download/foo-v1.2.0.dmg",
            verified: "example.com/download/"
        end
      CASK
    end

    include_examples "does not report any offenses"
  end

  context "when the url 'verified' value has a path component that doesn't end with a /" do
    let(:source) do
      <<~CASK
        cask "foo" do
          url "https://example.com/download/foo-v1.2.0.dmg",
            verified: "example.com/download"
        end
      CASK
    end

    let(:expected_offenses) do
      [{
        message:  "Verified URL parameter value should end with a /.",
        severity: :convention,
        line:     3,
        column:   14,
        source:   "\"example.com/download\"",
      }]
    end

    let(:correct_source) do
      <<~CASK
        cask "foo" do
          url "https://example.com/download/foo-v1.2.0.dmg",
            verified: "example.com/download/"
        end
      CASK
    end

    include_examples "reports offenses"
    include_examples "autocorrects source"
  end
end
