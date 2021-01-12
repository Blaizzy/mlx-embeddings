# typed: false
# frozen_string_literal: true

require "completions"

describe Completions do
  let(:internal_path) { HOMEBREW_REPOSITORY/"Library/Taps/homebrew/homebrew-bar" }
  let(:external_path) { HOMEBREW_REPOSITORY/"Library/Taps/foo/homebrew-bar" }

  before do
    HOMEBREW_REPOSITORY.cd do
      system "git", "init"
    end
    internal_path.mkpath
    external_path.mkpath
  end

  def setup_completions(external:)
    (internal_path/"completions/bash/foo_internal").write "#foo completions"
    if external
      (external_path/"completions/bash/foo_external").write "#foo completions"
    elsif (external_path/"completions/bash/foo_external").exist?
      (external_path/"completions/bash/foo_external").delete
    end
  end

  def setup_completions_setting(state, setting: "linkcompletions")
    HOMEBREW_REPOSITORY.cd do
      system "git", "config", "--replace-all", "homebrew.#{setting}", state.to_s
    end
  end

  def read_completions_setting(setting: "linkcompletions")
    HOMEBREW_REPOSITORY.cd do
      Utils.popen_read("git", "config", "--get", "homebrew.#{setting}").chomp.presence
    end
  end

  def delete_completions_setting(setting: "linkcompletions")
    HOMEBREW_REPOSITORY.cd do
      system "git", "config", "--unset-all", "homebrew.#{setting}"
    end
  end

  after do
    FileUtils.rm_rf internal_path
    FileUtils.rm_rf external_path.dirname
  end

  describe ".link!" do
    it "sets homebrew.linkcompletions to true" do
      setup_completions_setting false
      expect { described_class.link! }.not_to raise_error
      expect(read_completions_setting).to eq "true"
    end

    it "sets homebrew.linkcompletions to true if unset" do
      delete_completions_setting
      expect { described_class.link! }.not_to raise_error
      expect(read_completions_setting).to eq "true"
    end

    it "keeps homebrew.linkcompletions set to true" do
      setup_completions_setting true
      expect { described_class.link! }.not_to raise_error
      expect(read_completions_setting).to eq "true"
    end
  end

  describe ".unlink!" do
    it "sets homebrew.linkcompletions to false" do
      setup_completions_setting true
      expect { described_class.unlink! }.not_to raise_error
      expect(read_completions_setting).to eq "false"
    end

    it "sets homebrew.linkcompletions to false if unset" do
      delete_completions_setting
      expect { described_class.unlink! }.not_to raise_error
      expect(read_completions_setting).to eq "false"
    end

    it "keeps homebrew.linkcompletions set to false" do
      setup_completions_setting false
      expect { described_class.unlink! }.not_to raise_error
      expect(read_completions_setting).to eq "false"
    end
  end

  describe ".link_completions?" do
    it "returns true if homebrew.linkcompletions is true" do
      setup_completions_setting true
      expect(described_class.link_completions?).to be true
    end

    it "returns false if homebrew.linkcompletions is false" do
      setup_completions_setting false
      expect(described_class.link_completions?).to be false
    end

    it "returns false if homebrew.linkcompletions is not set" do
      expect(described_class.link_completions?).to be false
    end
  end

  describe ".completions_to_link?" do
    it "returns false if only internal taps have completions" do
      setup_completions external: false
      expect(described_class.completions_to_link?).to be false
    end

    it "returns true if external taps have completions" do
      setup_completions external: true
      expect(described_class.completions_to_link?).to be true
    end
  end

  describe ".show_completions_message_if_needed" do
    it "doesn't show the message if there are no completions to link" do
      setup_completions external: false
      delete_completions_setting setting: :completionsmessageshown
      expect { described_class.show_completions_message_if_needed }.not_to output.to_stdout
    end

    it "doesn't show the message if there are completions to link but the message has already been shown" do
      setup_completions external: true
      setup_completions_setting true, setting: :completionsmessageshown
      expect { described_class.show_completions_message_if_needed }.not_to output.to_stdout
    end

    it "shows the message if there are completions to link and the message hasn't already been shown" do
      setup_completions external: true
      delete_completions_setting setting: :completionsmessageshown

      # This will fail because the method calls `puts`.
      # If we output the `ohai` andcatch the error, we can be usre that the message is showing.
      error_message = "private method `puts' called for Completions:Module"
      expect { described_class.show_completions_message_if_needed }
        .to output.to_stdout
        .and raise_error(NoMethodError, error_message)
    end
  end
end
