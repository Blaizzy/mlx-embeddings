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
          cask 'foo' do
            postflight do
              foobar
            end
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

  context "when auditing `sha256` stanzas inside on_arch blocks" do
    context "when there are no on_arch blocks" do
      let(:source) do
        <<-CASK.undent
          cask 'foo' do
            sha256 "67cdb8a02803ef37fdbf7e0be205863172e41a561ca446cd84f0d7ab35a99d94"
          end
        CASK
      end

      include_examples "does not report any offenses"
    end

    context "when the proper `sha256` stanza is used" do
      let(:source) do
        <<-CASK.undent
          cask 'foo' do
            sha256 arm:   "67cdb8a02803ef37fdbf7e0be205863172e41a561ca446cd84f0d7ab35a99d94",
                   intel: "8c62a2b791cf5f0da6066a0a4b6e85f62949cd60975da062df44adf887f4370b"
          end
        CASK
      end

      include_examples "does not report any offenses"
    end

    context "when the `sha256` stanza needs to be removed from the on_arch blocks" do
      let(:source) do
        <<-CASK.undent
          cask 'foo' do
            on_intel do
              sha256 "67cdb8a02803ef37fdbf7e0be205863172e41a561ca446cd84f0d7ab35a99d94"
            end
            on_arm do
              sha256 "8c62a2b791cf5f0da6066a0a4b6e85f62949cd60975da062df44adf887f4370b"
            end
          end
        CASK
      end
      let(:correct_source) do
        <<-CASK.undent
          cask 'foo' do
          #{"  "}
            sha256 arm: "8c62a2b791cf5f0da6066a0a4b6e85f62949cd60975da062df44adf887f4370b", intel: "67cdb8a02803ef37fdbf7e0be205863172e41a561ca446cd84f0d7ab35a99d94"
          end
        CASK
      end
      let(:offense_source) do
        <<-CASK.undent
          on_arm do
              sha256 "8c62a2b791cf5f0da6066a0a4b6e85f62949cd60975da062df44adf887f4370b"
            end
        CASK
      end
      let(:expected_offenses) do
        [{
          message:  'Use `sha256 arm: "8c62a2b791cf5f0da6066a0a4b6e85f62949cd60975da062df44adf887f4370b", ' \
                    'intel: "67cdb8a02803ef37fdbf7e0be205863172e41a561ca446cd84f0d7ab35a99d94"` instead of ' \
                    "nesting the `sha256` stanzas in `on_intel` and `on_arm` blocks",
          severity: :convention,
          line:     5,
          column:   2,
          source:   offense_source.strip,
        }]
      end

      include_examples "reports offenses"

      include_examples "autocorrects source"
    end

    context "when there is only one on_arch block" do
      let(:source) do
        <<-CASK.undent
          cask 'foo' do
            on_intel do
              sha256 "67cdb8a02803ef37fdbf7e0be205863172e41a561ca446cd84f0d7ab35a99d94"
            end
          end
        CASK
      end

      include_examples "does not report any offenses"
    end

    context "when there is also a `version` stanza inside the on_arch blocks" do
      let(:source) do
        <<-CASK.undent
          cask 'foo' do
            on_intel do
              version "1.0.0"
              sha256 "67cdb8a02803ef37fdbf7e0be205863172e41a561ca446cd84f0d7ab35a99d94"
            end
            on_arm do
              version "2.0.0"
              sha256 "8c62a2b791cf5f0da6066a0a4b6e85f62949cd60975da062df44adf887f4370b"
            end
          end
        CASK
      end

      include_examples "does not report any offenses"
    end

    context "when there is also a `version` stanza inside only a single on_arch block" do
      let(:source) do
        <<-CASK.undent
          cask 'foo' do
            on_intel do
              version "2.0.0"
              sha256 "67cdb8a02803ef37fdbf7e0be205863172e41a561ca446cd84f0d7ab35a99d94"
            end
            on_arm do
              sha256 "8c62a2b791cf5f0da6066a0a4b6e85f62949cd60975da062df44adf887f4370b"
            end
          end
        CASK
      end

      include_examples "does not report any offenses"
    end
  end
end
