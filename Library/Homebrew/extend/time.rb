# typed: false
# frozen_string_literal: true

module TimeRemaining
  refine Time do
    def remaining
      [0, self - Time.now].max
    end

    def remaining!
      r = remaining

      raise Timeout::Error if r <= 0

      r
    end
  end
end
