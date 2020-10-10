# typed: false
# frozen_string_literal: true

require_relative "shared_examples/invalid_option"

describe Cask::Cmd::Home, :cask do
  before do
    allow(described_class).to receive(:open_url)
  end

  it_behaves_like "a command that handles invalid options"
end
