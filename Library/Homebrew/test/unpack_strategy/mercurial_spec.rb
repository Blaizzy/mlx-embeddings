# typed: false
# frozen_string_literal: true

require_relative "shared_examples"

describe UnpackStrategy::Mercurial do
  let(:repo) {
    mktmpdir.tap do |repo|
      (repo/".hg").mkpath
    end
  }
  let(:path) { repo }

  include_examples "UnpackStrategy::detect"
end
