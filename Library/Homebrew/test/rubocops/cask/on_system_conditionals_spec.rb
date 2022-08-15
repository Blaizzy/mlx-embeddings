# typed: false
# frozen_string_literal: true

require "rubocops/rubocop-cask"
require "test/rubocops/cask/shared_examples/cask_cop"

describe RuboCop::Cop::Cask::OnSystemConditionals do
  include CaskCop

  subject(:cop) { described_class.new }

  context "when auditing `postflight` stanzas" do
    context "when there are no on_system blocks" do
      let(:source) do
        <<-CASK.undent
          postflight do
            foobar
          end
        CASK
      end

      include_examples "does not report any offenses"
    end

    context "when there is an `on_intel` block" do
      let(:source) do
        <<-CASK.undent
          cask 'foo' do
            postflight do
              on_intel do
                foobar
              end
            end
          end
        CASK
      end
      let(:correct_source) do
        <<-CASK.undent
          cask 'foo' do
            postflight do
              if Hardware::CPU.intel?
                foobar
              end
            end
          end
        CASK
      end
      let(:expected_offenses) do
        [{
          message:  "Don't use `on_intel` in `postflight do`, use `if Hardware::CPU.intel?` instead.",
          severity: :convention,
          line:     3,
          column:   4,
          source:   "on_intel",
        }]
      end

      include_examples "reports offenses"

      include_examples "autocorrects source"
    end

    context "when there is an `on_monterey` block" do
      let(:source) do
        <<-CASK.undent
          cask 'foo' do
            postflight do
              on_monterey do
                foobar
              end
            end
          end
        CASK
      end
      let(:correct_source) do
        <<-CASK.undent
          cask 'foo' do
            postflight do
              if MacOS.version == :monterey
                foobar
              end
            end
          end
        CASK
      end
      let(:expected_offenses) do
        [{
          message:  "Don't use `on_monterey` in `postflight do`, use `if MacOS.version == :monterey` instead.",
          severity: :convention,
          line:     3,
          column:   4,
          source:   "on_monterey",
        }]
      end

      include_examples "reports offenses"

      include_examples "autocorrects source"
    end

    context "when there is an `on_monterey :or_older` block" do
      let(:source) do
        <<-CASK.undent
          cask 'foo' do
            postflight do
              on_monterey :or_older do
                foobar
              end
            end
          end
        CASK
      end
      let(:correct_source) do
        <<-CASK.undent
          cask 'foo' do
            postflight do
              if MacOS.version <= :monterey
                foobar
              end
            end
          end
        CASK
      end
      let(:expected_offenses) do
        [{
          message:  "Don't use `on_monterey :or_older` in `postflight do`, " \
                    "use `if MacOS.version <= :monterey` instead.",
          severity: :convention,
          line:     3,
          column:   4,
          source:   "on_monterey :or_older",
        }]
      end

      include_examples "reports offenses"

      include_examples "autocorrects source"
    end
  end
end
