# typed: false
# frozen_string_literal: true

require "rubocops/rubocop-cask"
require "test/rubocops/cask/shared_examples/cask_cop"

describe RuboCop::Cop::Cask::NoOverrides do
  include CaskCop

  subject(:cop) { described_class.new }

  context "when there are no on_system blocks" do
    let(:source) do
      <<~CASK
        cask 'foo' do
          version '1.2.3'
          url 'https://brew.sh/foo.pkg'

          name 'Foo'
        end
      CASK
    end

    include_examples "does not report any offenses"
  end

  context "when there are no top-level standalone stanzas" do
    let(:source) do
      <<~CASK
        cask 'foo' do
          on_mojave :or_later do
            version :latest
          end
        end
      CASK
    end

    include_examples "does not report any offenses"
  end

  context "when there's only one difference between the `on_*` blocks" do
    let(:source) do
      <<~CASK
        cask "foo" do
          version "1.2.3"

          on_big_sur :or_older do
            sha256 "bbb"
            url "https://brew.sh/legacy/foo-2.3.4.dmg"
          end
          on_monterey :or_newer do
            sha256 "aaa"
            url "https://brew.sh/foo-2.3.4.dmg"
          end
        end
      CASK
    end

    include_examples "does not report any offenses"
  end

  context "when there are multiple differences between the `on_*` blocks" do
    let(:source) do
      <<~CASK
        cask "foo" do
          version "1.2.3"
          sha256 "aaa"
          url "https://brew.sh/foo-2.3.4.dmg"

          on_big_sur :or_older do
            sha256 "bbb"
            url "https://brew.sh/legacy/foo-2.3.4.dmg"
          end
        end
      CASK
    end

    let(:expected_offenses) do
      [{
        message:  <<~EOS,
          Do not use a top-level `sha256` stanza as the default. Add it to an `on_{system}` block instead.
          Use `:or_older` or `:or_newer` to specify a range of macOS versions.
        EOS
        severity: :convention,
        line:     3,
        column:   2,
        source:   "sha256 \"aaa\"",
      }, {
        message:  <<~EOS,
          Do not use a top-level `url` stanza as the default. Add it to an `on_{system}` block instead.
          Use `:or_older` or `:or_newer` to specify a range of macOS versions.
        EOS
        severity: :convention,
        line:     4,
        column:   2,
        source:   "url \"https://brew.sh/foo-2.3.4.dmg\"",
      }]
    end

    include_examples "reports offenses"
  end

  context "when there are top-level standalone stanzas" do
    let(:source) do
      <<~CASK
        cask 'foo' do
          version '2.3.4'
          on_mojave :or_older do
            version '1.2.3'
          end

          url 'https://brew.sh/foo-2.3.4.dmg'
        end
      CASK
    end

    let(:expected_offenses) do
      [{
        message:  <<~EOS,
          Do not use a top-level `version` stanza as the default. Add it to an `on_{system}` block instead.
          Use `:or_older` or `:or_newer` to specify a range of macOS versions.
        EOS
        severity: :convention,
        line:     2,
        column:   2,
        source:   "version '2.3.4'",
      }]
    end

    include_examples "reports offenses"
  end
end
