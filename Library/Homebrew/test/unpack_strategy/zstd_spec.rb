# typed: false
# frozen_string_literal: true

require_relative "shared_examples"

describe UnpackStrategy::Zstd do
  let(:path) { TEST_FIXTURE_DIR/"cask/container.tar.zst" }

  include_examples "UnpackStrategy::detect"
end
