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
              verified: "example.com/download/"
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
              verified: "https://example.com/download/"
        end
      CASK
    end

    let(:expected_offenses) do
      [{
        message:  "Verified URL parameter value should not contain a URL scheme.",
        severity: :convention,
        line:     3,
        column:   16,
        source:   "\"https://example.com/download/\"",
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

  context "when url 'verified' value does not have a path component" do
    context "when the URL ends with a slash" do
      let(:source) do
        <<~CASK
          cask "foo" do
            url "https://example.org/",
                verified: "example.org/"
          end
        CASK
      end

      include_examples "does not report any offenses"
    end

    context "when the URL does not end with a slash" do
      let(:source) do
        <<~CASK
          cask "foo" do
            url "https://example.org/",
                verified: "example.org"
          end
        CASK
      end

      let(:expected_offenses) do
        [{
          message:  "Verified URL parameter value should end with a /.",
          severity: :convention,
          line:     3,
          column:   16,
          source:   "\"example.org\"",
        }]
      end

      let(:correct_source) do
        <<~CASK
          cask "foo" do
            url "https://example.org/",
                verified: "example.org/"
          end
        CASK
      end

      include_examples "reports offenses"
      include_examples "autocorrects source"
    end
  end

  context "when the URL does not end with a slash" do
    let(:source) do
      <<~CASK
        cask "foo" do
          url "https://github.com/Foo",
              verified: "github.com/Foo"
        end
      CASK
    end

    include_examples "does not report any offenses"
  end

  context "when the url ends with a / and the verified value does too" do
    let(:source) do
      <<~CASK
        cask "foo" do
          url "https://github.com/",
              verified: "github.com/"
        end
      CASK
    end

    include_examples "does not report any offenses"
  end

  context "when the url ends with a / and the verified value does not" do
    let(:source) do
      <<~CASK
        cask "foo" do
          url "https://github.com/",
              verified: "github.com"
        end
      CASK
    end

    let(:expected_offenses) do
      [{
        message:  "Verified URL parameter value should end with a /.",
        severity: :convention,
        line:     3,
        column:   16,
        source:   "\"github.com\"",
      }]
    end

    let(:correct_source) do
      <<~CASK
        cask "foo" do
          url "https://github.com/",
              verified: "github.com/"
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
          url "https://github.com/Foo/foo/releases/download/v1.2.0/foo-v1.2.0.dmg",
              verified: "github.com/Foo/foo/"
        end
      CASK
    end

    include_examples "does not report any offenses"
  end

  context "when the url has interpolation in it and the verified url ends with a /" do
    let(:source) do
      <<~CASK
        cask "foo" do
          version "1.2.3"
          url "https://example.com/download/foo-v\#{version}.dmg",
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
          url "https://github.com/Foo/foo/releases/download/v1.2.0/foo-v1.2.0.dmg",
              verified: "github.com/Foo/foo"
        end
      CASK
    end

    let(:expected_offenses) do
      [{
        message:  "Verified URL parameter value should end with a /.",
        severity: :convention,
        line:     3,
        column:   16,
        source:   "\"github.com/Foo/foo\"",
      }]
    end

    let(:correct_source) do
      <<~CASK
        cask "foo" do
          url "https://github.com/Foo/foo/releases/download/v1.2.0/foo-v1.2.0.dmg",
              verified: "github.com/Foo/foo/"
        end
      CASK
    end

    include_examples "reports offenses"
    include_examples "autocorrects source"
  end
end
